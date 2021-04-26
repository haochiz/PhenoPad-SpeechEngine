#!/usr/bin/env python
#
# Copyright 2013 Tanel Alumae

"""
Reads speech data via websocket requests, sends it to Redis, waits for results from Redis and
forwards to client via websocket
"""
import os
import sys
import logging
import json
import codecs
import os.path
import uuid
import time
import threading
import functools
import struct
import socket
import traceback
import subprocess
from Queue import Queue
import multiprocessing
from multiprocessing import Process
from psutil import process_iter
from signal import SIGTERM

import ws4py.messaging

import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import tornado.gen
import tornado.concurrent
import concurrent.futures
import settings
import common
import numpy

import datetime

import odas_sst_worker as sst_worker
import speaker_embedding_worker as embedding_worker

import webrtcvad
import collections
import wave

DATA_PATH = './data/'

class Application(tornado.web.Application):
    def __init__(self):
        settings = dict(
            cookie_secret="43oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=",
            template_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
            static_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
            xsrf_cookies=False,
            autoescape=None,
        )

        handlers = [
            (r"/", MainHandler),
            (r"/client/ws/speech", DecoderSocketHandler),
            (r"/client/ws/status", StatusSocketHandler),
            (r"/client/ws/speech_result", SpeechResultSocketHandler),
            (r"/client/dynamic/reference", ReferenceHandler),
            (r"/client/dynamic/recognize", HttpChunkedRecognizeHandler),
            (r"/worker/ws/speech", WorkerSocketHandler),
            (r"/client/static/(.*)", tornado.web.StaticFileHandler, {'path': settings["static_path"]}),
            # test
            (r"/client/ws/file_request", FileRequestSocketHandler)   
        ]
        tornado.web.Application.__init__(self, handlers, **settings)
        self.available_workers = set()
        self.status_listeners = set()
        self.num_requests_processed = 0
    
        self.speech_result_clients = {} # clients that receive ASR results 
        self.decoder_handlers = {} # keeps track of decoder handlers

    def send_status_update_single(self, ws):
        status = dict(num_workers_available=len(self.available_workers), num_requests_processed=self.num_requests_processed)
        print('##### printing at status update #####')
        print(status)
        ws.write_message(json.dumps(status))

    def send_status_update(self):
        for ws in self.status_listeners:
            print('##### ws: ', ws)
            self.send_status_update_single(ws)

    def save_reference(self, content_id, content):
        refs = {}
        try:
            with open("reference-content.json") as f:
                refs = json.load(f)
        except:
            pass
        refs[content_id] = content
        with open("reference-content.json", "w") as f:
            json.dump(refs, f, indent=2)



class MainHandler(tornado.web.RequestHandler):
    def get(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.join(current_directory, os.pardir)
        readme = os.path.join(parent_directory, "README.md")
        self.render(readme)



def content_type_to_caps(content_type):
    """
    Converts MIME-style raw audio content type specifier to GStreamer CAPS string
    """
    default_attributes= {"rate": 16000, "format" : "S16LE", "channels" : 1, "layout" : "interleaved"}
    media_type, _, attr_string = content_type.replace(";", ",").partition(",")
    if media_type in ["audio/x-raw", "audio/x-raw-int"]:
        media_type = "audio/x-raw"
        attributes = default_attributes
        for (key,_,value) in [p.partition("=") for p in attr_string.split(",")]:
            attributes[key.strip()] = value.strip()
        return "%s, %s" % (media_type, ", ".join(["%s=%s" % (key, value) for (key,value) in attributes.iteritems()]))
    else:
        return content_type



@tornado.web.stream_request_body
class HttpChunkedRecognizeHandler(tornado.web.RequestHandler):
    """
    Provides a HTTP POST/PUT interface supporting chunked transfer requests, similar to that provided by
    http://github.com/alumae/ruby-pocketsphinx-server.
    """

    def prepare(self):
        self.id = str(uuid.uuid4())
        self.final_hyp = ""
        self.final_result_queue = Queue()
        self.user_id = self.request.headers.get("device-id", "none")
        self.content_id = self.request.headers.get("content-id", "none")
        logging.info("%s: OPEN: user='%s', content='%s'" % (self.id, self.user_id, self.content_id))
        self.worker = None
        self.error_status = 0
        self.error_message = None
        #Waiter thread for final hypothesis:
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 
        try:
            self.worker = self.application.available_workers.pop()
            self.application.send_status_update()
            logging.info("%s: Using worker %s" % (self.id, self.__str__()))
            self.worker.set_client_socket(self)

            content_type = self.request.headers.get("Content-Type", None)
            if content_type:
                content_type = content_type_to_caps(content_type)
                logging.info("%s: Using content type: %s" % (self.id, content_type))

            self.worker.write_message(json.dumps(dict(id=self.id, \
        content_type=content_type, user_id=self.user_id, content_id=self.content_id)))
        except KeyError:
            logging.warn("%s: No worker available for client request" % self.id)
            self.set_status(503)
            self.finish("No workers available")

    def data_received(self, chunk):
        assert self.worker is not None
        logging.debug("%s: Forwarding client message of length %d to worker" % (self.id, len(chunk)))
        self.worker.write_message(chunk, binary=True)

    def post(self, *args, **kwargs):
        self.end_request(args, kwargs)

    def put(self, *args, **kwargs):
        self.end_request(args, kwargs)

    @tornado.concurrent.run_on_executor
    def get_final_hyp(self):
        logging.info("%s: Waiting for final result..." % self.id)
        return self.final_result_queue.get(block=True)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def end_request(self, *args, **kwargs):
        logging.info("%s: Handling the end of chunked recognize request" % self.id)
        assert self.worker is not None
        self.worker.write_message("EOS", binary=True)
        logging.info("%s: yielding..." % self.id)
        hyp = yield self.get_final_hyp()
        if self.error_status == 0:
            logging.info("%s: Final hyp: %s" % (self.id, hyp))
            response = {"status" : 0, "id": self.id, "hypotheses": [{"utterance" : hyp}]}
            self.write(response)
        else:
            logging.info("%s: Error (status=%d) processing HTTP request: %s" % (self.id, self.error_status, self.error_message))
            response = {"status" : self.error_status, "id": self.id, "message": self.error_message}
            self.write(response)
        self.application.num_requests_processed += 1
        self.application.send_status_update()
        self.worker.set_client_socket(None)
        self.worker.close()
        self.finish()
        logging.info("Everything done")

    def send_event(self, event):
        event_str = str(event)
        if len(event_str) > 100:
            event_str = event_str[:97] + "..."
        logging.info("%s: Receiving event %s from worker" % (self.id, event_str))
        if event["status"] == 0 and ("result" in event):
            try:
                if len(event["result"]["hypotheses"]) > 0 and event["result"]["final"]:
                    if len(self.final_hyp) > 0:
                        self.final_hyp += " "
                    self.final_hyp += event["result"]["hypotheses"][0]["transcript"]
            except:
                e = sys.exc_info()[0]
                logging.warn("Failed to extract hypothesis from recognition result:" + e)
        elif event["status"] != 0:
            self.error_status = event["status"]
            self.error_message = event.get("message", "")

    def close(self):
        logging.info("%s: Receiving 'close' from worker" % (self.id))
        self.final_result_queue.put(self.final_hyp)




class ReferenceHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        content_id = self.request.headers.get("Content-Id")
        if content_id:
            content = codecs.decode(self.request.body, "utf-8")
            user_id = self.request.headers.get("User-Id", "")
            self.application.save_reference(content_id, dict(content=content, user_id=user_id, time=time.strftime("%Y-%m-%dT%H:%M:%S")))
            logging.info("Received reference text for content %s and user %s" % (content_id, user_id))
            self.set_header('Access-Control-Allow-Origin', '*')
        else:
            self.set_status(400)
            self.finish("No Content-Id specified")

    def options(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.set_header('Access-Control-Max-Age', 1000)
        # note that '*' is not valid for Access-Control-Allow-Headers
        self.set_header('Access-Control-Allow-Headers',  'origin, x-csrftoken, content-type, accept, User-Id, Content-Id')



class StatusSocketHandler(tornado.websocket.WebSocketHandler):
    # needed for Tornado 4.0
    def check_origin(self, origin):
        return True

    def open(self):
        logging.info("New status listener")
        self.application.status_listeners.add(self)
        self.application.send_status_update_single(self)

    def on_close(self):
        logging.info("Status listener left")
        self.application.status_listeners.remove(self)



class WorkerSocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        tornado.websocket.WebSocketHandler.__init__(self, application, request, **kwargs)
        self.client_socket = None
        self.result_client = False
        self.external = False 
        self.decoder_socket = None
        self.ftrans = None
        
        self.diarization_result_queue = None
        self.diarization_segment_queue = None

    # needed for Tornado 4.0
    def check_origin(self, origin):
        return True

    def open(self):
        self.client_socket = None
        self.application.available_workers.add(self)
        print('@-------@: worker added!!')
        logging.info("New worker available " + self.__str__())
        self.application.send_status_update()

    def on_close(self):
        logging.info("Worker " + self.__str__() + " leaving")
        self.application.available_workers.discard(self)
        try:
            if self.client_socket:
                self.client_socket.close()
            self.application.send_status_update()
            if self.ftrans:
                self.ftrans.close()
            if self.decoder_socket:
                self.decoder_socket.on_connection_close()
            if self.diarization_result_queue:
                self.diarization_result_queue.close()
                del self.diarization_result_queue
        except Exception, e:
            print('Warning: WorkerHandler on_close process cannot be completed')
            print(traceback.format_exc())

    def on_message(self, message):        
        assert self.client_socket is not None
        #print message #XXX testing 
        if message == "TIMEOUT":
            print("TIMEOUT")
            #self.client_socket.write_message("TIMEOUT_EXIT")
            self.on_close()
            return
        event = json.loads(message)
        # try reconnect to result_client if necessary #                                                           
        if self.external == True: self.try_reconnect_result_client()                                                                   
        # if connection to result_client lost #
        if not self.client_socket and self.result_client == True:   
            self.set_client_socket(self.decoder_socket)             
            self.result_client = False                              
        if 'result' in event:
            # if result is not final, add diarization results # 
            if not event['result']['final']:
                try:                                                    
                    event = self.get_diarization_result(event)          
                    #self.decoder_socket.send_event(event)              
                except Exception, e:                                                 
                    print('Warning: unable to pack partial result')
                    print(traceback.format_exc())

            # send ASR result to PhenoPad #                                                           
            self.client_socket.send_event(event)
        elif 'adaptation_state' in event:
            logging.info("Worker Handler got adaptation state")
            self.client_socket.send_event(event)
        else:
            print("WorkerHandler Received Event", event)
        # save transcript to a local file # 
        if 'result' in event:
            dt = datetime.datetime.now()
            if event['result']['final']:                                                          
                self.ftrans.write('%s:%s:%s  ' % (dt.hour,dt.minute,dt.second) + \
                                  str(event['result']['hypotheses'][0]['transcript'] + '\n'))                                                                                                  
    def set_client_socket(self, client_socket):
        self.client_socket = client_socket

    def try_reconnect_result_client(self):
        """ this function checks if the result client has not been connected while 
            using external microphone, if not, it checks if the result client is 
            connected now and set worker's client_socket to theresult client if it is
        """
        result_clients = self.application.speech_result_clients # {"manager_id": TODO?}
        request_uri = self.client_socket.request.uri
        try:
            manager_id = request_uri.split('manager_id=')[1] # str(three integers) for now; e.g. "666"
        except IndexError:
            # no manager id, set external to false
            self.external = False
        if self.result_client == False and self.external == True:
            print('try get: ', manager_id)
            if result_clients.get(manager_id) is not None:
                print('switch: ', result_clients)
                print(result_clients[manager_id])
                self.set_client_socket(result_clients[manager_id])
                print('switch client socket success: ', \
                      result_clients[manager_id])
                self.result_client = True

    def get_diarization_result(self, event):
        """ try to get diarization results from the diarization process
            and add the obtained results to the recognition result
        """
        diarization_results_list = []
        diarization_incremental = True
        apdx = None
        try:
            while not self.diarization_result_queue.empty():
                diarization_result, diarization_incremental = self.diarization_result_queue.get(True)
                if diarization_incremental:
                    diarization_results_list.append(diarization_result)
                else:
                    diarization_results_list = diarization_result
        except:
            print("WorkerHandler: fail to get diarization results from Queue")        
        event['result']['diarization'] = diarization_results_list
        event['result']['diarization_incremental'] = diarization_incremental
        if diarization_incremental == False:
            print 'master_server: received recluster results'

        return event



class SpeechResultSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        self.id = str(uuid.uuid4())
        self.client_id = None

        logging.info("ASR result server opened")
    
        print('result_socket.request.uri is: ', self.request.uri)
        self.client_id = self.request.uri.split('manager_id=')[1]
        self.application.speech_result_clients[self.client_id] = self

    def on_connection_close(self):
        logging.info("ASR result server closed")
        self.application.speech_result_clients[self.client_id] = None
        self = None

    def on_message(self, message):
        logging.info("ASR result server received: " + message)

    def send_event(self, event):
        event["id"] = self.id
        event_str = str(event)
        if len(event_str) > 100:
            event_str = event_str[:97] + "..."
        #logging.info("%s: Sending event %s to client" % (self.id, event_str))
        try:
            self.write_message(json.dumps(event))
        except: 
            logging.warning('Result client lost connection!')
            self = None



class FileRequestSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('request_socket.request.uri is: ', self.request.uri)
        

    def on_connection_close(self):
        logging.info("File request socket closed")
        print("File request socket closed")
        self = None

    def on_message(self, message):
        logging.info("File request socket received: " + message)
        print("File request socket received: " + message)
        # message = "<filename> <start> <end>"
        try:
            if len(message.split()) == 1:
                #TODO: send full file
                filename = message.split()[0]
                wav_name = self.get_full_audio(filename)
            else:
                # get filename, start, end from message
                filename = message.split()[0]
                start = float(message.split()[1])
                end = float(message.split()[2])
                wav_name = self.get_audio_seg(filename, start, end)
            self.send_audio(wav_name)
        
        except Exception, e:
            print(traceback.format_exc())

    def send_event(self, event):
        #logging.info("%s: Sending event %s to client" % (self.id, event_str))
        try:
            self.write_message(json.dumps(event))
        except:
            logging.warning('Result client lost connection!')
            self = None

    def send_audio(self, wav_name):
        """ send a temporary wave audio file to client, delete the 
            file after send complete
        """
        time.sleep(1)
        try:
            print("sending audio...")
            audio_file = open(wav_name, "rb")
            block = audio_file.read(200*1024)
            while block: 
                self.write_message(block, binary=True)
                block = audio_file.read(200*1024)
            print('audio sent')
            # delete temp wav file
            command = "rm " + wav_name
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        except Exception, e:
            print(traceback.format_exc())
            # if send failed, delete the temp audio file
            command = "rm " + wav_name
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

    def get_audio_seg(self, filename, start, end): 
        """ extarct a segment from an audio file and save it as a temp wav file,
            returns the filename of the temp file
            args:    str filename: unique identifier of the audiofile
                     float start: starting time of the requested seg in sec
                     float end: ending time of the requested seg in sec
        """
        path = DATA_PATH + '/audiofiles/single_channel/'
        # read audiodata from the file
        audiofile = open(path+filename+".raw", "rb") # NOTE: the exact format of the name is not finalized
        raw_audio = audiofile.read()
        # get the requested segment
        seg = raw_audio[int(start*32000):int(end*32000)] # each byte in the bytearray is 1/(16000*2) sec long
        # write as an wav file
        wav_name = path + filename + '_' + str(start) +  '_' + str(end) + '.wav'
        wave_write = wave.open(wav_name, "wb")
        wave_write.setparams((1, 2, 16000, (end-start)*32000, 'NONE', 'NONE')) # see python Wave document for args detail
        wave_write.writeframes(seg)
        
        return wav_name 

    def get_full_audio(self, filename):
        path = DATA_PATH + 'audiofiles/single_channel/'
        # read audiodata from the file
        audiofile = open(path+filename+".raw", "rb")
        raw_audio = audiofile.read()
        wav_name = path+filename+".wav"
        wave_write = wave.open(wav_name, "wb")
        wave_write.setparams((1, 2, 16000, len(raw_audio), 'NONE', 'NONE')) # see python Wave document for args detail
        wave_write.writeframes(raw_audio)

        return wav_name

class DecoderSocketHandler(tornado.websocket.WebSocketHandler):
    # needed for Tornado 4.0
    def check_origin(self, origin):
        return True

    def send_event(self, event):
        event["id"] = self.id
        event_str = str(event)
        if len(event_str) > 100:
            event_str = event_str[:97] + "..."
        logging.info("%s: Sending event %s to client" % (self.id, event_str))

        self.write_message(json.dumps(event))

    def open(self):
        self.id = str(uuid.uuid4())
        logging.info("%s: OPEN" % (self.id))
        logging.info("%s: Request arguments: %s" % \
                     (self.id, " ".join(["%s=\"%s\"" % \
                     (a, self.get_argument(a)) \
                     for a in self.request.arguments])))
        
        self.key = str(len(self.application.decoder_handlers))
        self.application.decoder_handlers[self.key] = self
        print(self.application.decoder_handlers)
        self.user_id = self.get_argument("user-id", "none", True)
        self.content_id = self.get_argument("content-id", "none", True)
        
        self.worker = None
        self.client_id = None
        self.external = False
        self.DECODER_STATUS = "OPEN"
        self.segmentation_method = "VAD" # choose between ODAS and VAD; NOTE: ODAS is not available right now

        if self.segmentation_method == "ODAS":
            # ODAS related attributes
            self.odas_sock = None
            self.odas_conn = None
            self.sst_sock = None
            self.sst_conn = None
            self.odas_pid = None
            self.odas_process = None
            self.odas_cfgs = ['a', 'b', 'c', 'd'] 
            # ODAS management
            self.restarting_odas = False
            self.odas_dead = False # temp 
            self.odas_audio_queue = multiprocessing.Queue()
        self.sst_queue = multiprocessing.Queue()

        # VAD related attributes
        self.speech_start = time.time()
        self.vad = webrtcvad.Vad(3)
        self.dps_buffer = []
        self.vad_buffer = []
        self.triggered = False
        self.current_seg_start = None
        self.frame_count = 0
        self.num_frame = 0
        self.ring_buffer = collections.deque(maxlen=50)

        # test
        self.tmp_audio_buffer = []

        # TODO: "manager_id" may need some rework
        self.set_manager_id() # if request.uri contains manager_id,
                              # set client_id and self.external=True
        
        # Audiofile related attributs
        self.audio_name = str(self.request.uri.split('audio_name=')[1])
        print(self.audio_name)
        self.temp_audio_file = open('./temp_audio_file.raw', 'wb')
        self.singlechannel_audio_file = open(DATA_PATH + 'audiofiles/single_channel/' + self.audio_name + '.raw', 'a+b')
        if self.external:
            self.multichannel_audio_file = open(DATA_PATH + 'audiofiles/multi_channel/' + self.audio_name + '.raw', 'a+b')
        
        self.setup_asr_worker()
        
        # start the diarization process(es)
        if self.external == True and self.segmentation_method == "ODAS":
            self.start_odas_process()
        self.start_diarization_process()                              
        

    def on_connection_close(self):
        print("Decoder Handler: on connection close")
        logging.info("%s: Handling on_connection_close()" % self.id)
        self.application.num_requests_processed += 1
        self.application.send_status_update()

        self.DECODER_STATUS = "CLOSE"
        
        if self.worker:
            try:
                self.worker.set_client_socket(None)
                logging.info("%s: Closing worker connection" % self.id)
                self.worker.close()
            except:
                pass

        if self.external == True:
            # close sst result transcript file
            self.sst_trans.close()
            print("sst_trans closed")
        # Kill Odas and diarization process
        self.close_diarization_process()
        # close single-channel audio file
        self.singlechannel_audio_file.close()
        # close multi-channel audio file
        self.multichannel_audio_file.close()
        # close embedding temp audio file
        self.temp_audio_file.close()
        try:
            self.embedding_process.terminate()
        except:
            pass
    
    def on_message(self, message):
        # assert self.worker is not None
        if self.worker == None:
            print("No Worker found, closing...")
            self.on_connection_close()
        #logging.info("%s: Forwarding client message (%s) of length %d to worker" % \
        #             (self.id, type(message), len(message)))
        try:
            if self.DECODER_STATUS == "OPEN":
                if isinstance(message, unicode):
                    self.worker.write_message(message, binary=False)
                else:
                    # if not using external microphone (RPi)
                    if self.external == False:

                        self.worker.write_message(message, binary=True)
                        logging.info("%s: Forwarding client message (%s) of length %d to worker" % \
                                        (self.id, type(message), len(message)))
                        
                        # convert float32 audio to s-int16 audio
                        message = self.convert_f32_to_s16(message)
                        # save temproray audio for embedding worker
                        self.temp_audio_file.write(message)
                        self.temp_audio_file.flush()
                        # save single channel raw audio
                        self.singlechannel_audio_file.write(message)
                        self.singlechannel_audio_file.flush()
                        # VAD segmentation
                        self.vad_segment(message)
                    else:
                        # send mono audio to kaldi worker #
                        message_short = self.audio_raw_extract_first_channel(message)
                        self.worker.write_message(message_short, binary=True)
                        logging.info("%s: Forwarding client message (%s) of length %d to worker" % \
                                        (self.id, type(message), len(message)))
                        # save temproray audio for embedding worker
                        self.temp_audio_file.write(message_short)
                        self.temp_audio_file.flush()
                        # save multi channel and single channel raw audio
                        self.multichannel_audio_file.write(message)
                        self.multichannel_audio_file.flush()
                        self.singlechannel_audio_file.write(message_short)
                        self.singlechannel_audio_file.flush()

                        if self.segmentation_method == "ODAS": 
                            # send audio to and receive sst from odas #
                            # TODO: get rid of this, if ODAS is gone just switch to VAD segmentation
                            if not self.odas_dead and self.restarting_odas == False:
                                #self.send_audio_to_odas(self.amplify_raw_audio(message))
                                self.send_audio_to_odas(message)
                                self.recv_sst_from_odas()

                        """ currently VAD segmentation is always running, when using ODAS the final result
                            is not processed and displayed
                        """
                        if self.segmentation_method != None:  
                            # VAD segmentation
                            self.vad_segment_external(message_short)
        except Exception, e:
            # TODO: handle this part better
            print("Decoder Handler: Cannot forward message")
            print(traceback.format_exc())

    def setup_asr_worker(self):
        try:
            # set up kaldi worker
            self.worker = self.application.available_workers.pop()                    
            print("DecoderHandler: worker popped")                                    
            self.application.send_status_update()                                     
            logging.info("%s: Using worker %s" % \
                         (self.id, self.__str__()))                                                                                 
            self.worker.decoder_socket = self                                         
            self.worker.external = self.external                                                                          
            dt = datetime.datetime.now()
            #TODO: rework ASR transcripts, use audio name instead
            self.worker.ftrans = open(DATA_PATH + 'saved_kaldi_asr_results/' + 'transcript_' + '%s_%s_%s_%s_%s_%s' % \
                                      (self.client_id,dt.year,dt.month,dt.day,dt.hour,dt.minute) + \
                                      '.txt', "a+")                                   
            # set client socket for worker
            self.set_worker_client_socket()                                           
            # send initialization message to worker #                          
            content_type = self.get_argument("content-type", None, True)              
            if content_type:                                                          
                logging.info("%s: Using content type: %s" % (self.id, content_type))  
            self.worker.write_message(json.dumps(dict(id = self.id, \
                                                      content_type = content_type, \
                                                      user_id = self.user_id, \
                                                      content_id = self.content_id)))
        except KeyError:
            logging.warn("%s: No worker available for client request" % self.id)
            event = dict(status=common.STATUS_NOT_AVAILABLE, \
                         message = "No decoder available, try again later")
            self.send_event(event)
            self.on_conncetion_close() 

    def set_manager_id(self):
        """ if manager_id in request uri, RPI/external microphone is used, 
            then record the decoder client id (which is the unique 
            identification of each SurfaceClient-ExternalMicrophone pair) 
            and set external (for external mic) as True 
        """
        if self.request.uri.find('manager_id') >= 0:
            self.client_id = self.request.uri.split('manager_id=')[1][:3]
            print('decoder.clientid is: ', self.client_id)
            self.external = True

    def set_worker_client_socket(self):
        """ Try connecting to the result client, if the connection cannot be connected,
            try again after one seconds for maximum 3 times
        """
        if self.external == False:
            print("using surface microphone")
            self.worker.set_client_socket(self)
            return
        if not self.application.speech_result_clients.get(self.client_id):
            print('Hello? I cannot find result client!')
            for i in range(3):
                if not self.application.speech_result_clients.get(self.client_id):
                    print('keep looking')
                    time.sleep(1)
                else:
                    self.worker.set_client_socket(self.application.speech_result_clients[self.client_id])
                    self.worker.result_client = True
                    break
        else:
            self.worker.set_client_socket(self.application.speech_result_clients[self.client_id])
            self.worker.result_client = True

        # if cannot connect to result client, set worker's client_socket as self for now 
        if not self.application.speech_result_clients.get(self.client_id):
            logging.info("Cannot find result client, setting client socket to self instead")
            print("Decoder cannot find result client, setting client socket to self instead")
            self.worker.set_client_socket(self)    


    ### audio related functions ###
    def audio_raw_average_channels(self, data):
        """ This takes a block of audio data (for now assume chunck size
            of 1024 bytes per channel) with n-channels and returns a single
            -channel data which is the average of all n channels
        """
        #TODO: add more detailed documentation on why it's implemented this way
        chunck_size = 1024 # for now
        #print(len(data))
        n_channel = (len(data)/2)/(chunck_size)
        if n_channel == 0:
            return data
        data_unpack_list = list(struct.unpack("%dh"%(len(data)/2), data))
        data_short_unpack_list_ = []
        data_short_unpack_list = []
        for n in range(n_channel): 
            temp_data_list = [] 
            for i in range(chunck_size):
                temp_data_list.append(data_unpack_list[i*n_channel+n])
            data_short_unpack_list_.append(temp_data_list)
        for i in range(chunck_size): 
            temp_sum = 0
            for n in range(n_channel):
                temp_sum = temp_sum + data_short_unpack_list_[n][i]
            data_short_unpack_list.append(temp_sum//n_channel)
        data_short_unpack = tuple(data_short_unpack_list)
        data_short = struct.pack("%dh"%(chunck_size), *data_short_unpack)
        
        return data_short

    def audio_raw_extract_first_channel(self, data):
        """ Extract the first channel (for now assume chunck size of 1024 bytes 
            per channel) from an n-channel multichannel audio
        """
        chunck_size = 1024 # for now
        n_channel = (len(data)/2)/(chunck_size)
        if n_channel < 1:
            # raise a warning as this should not happen
            logging.warn("Insufficient audio size, computed number of channels smaller than 1")
            return data

        elif n_channel == 1:
            return data
        else: 
            data_unpack_list = list(struct.unpack("%dh"%(len(data)/2), data))
            #data_short_unpack_list_ = []
            data_short_unpack_list = []
        for i in range(chunck_size):
            data_short_unpack_list.append(data_unpack_list[i*n_channel + 0])
        data_short_unpack = tuple(data_short_unpack_list)
        data_short = struct.pack("%dh"%(chunck_size), *data_short_unpack)
    
        return data_short

    def amplify_raw_audio(self, raw_audio):
        """ This is a test function written to amplify the raw audio from
            the microphone array, since the original audio's is not loud 
            enough
        """
        try:
            amp_factor = 1.15
            audio_data = numpy.fromstring(raw_audio, numpy.int16)
            amplified_audio_data = audio_data * amp_factor
            amplified_audio = struct.pack('h'*len(amplified_audio_data),*amplified_audio_data)
            return amplified_audio
        except:
            return raw_audio

    def convert_f32_to_s16(self, raw_audio_f32):
        """ Converts float32 audio from PhenoPad app to signed int16
            audio for VAD
        """
        try:
            # multiply by 32767 because data in range [-1.0f,+1.0f]
            audio_data_f32 = numpy.fromstring(raw_audio_f32, numpy.float32)*32767
            audio_data_s16 = audio_data_f32.astype(numpy.int16)
            raw_audio_s16 = struct.pack('h'*len(audio_data_s16),*audio_data_s16)
            return raw_audio_s16
        except:
            return raw_audio_f32


    ### Odas related functions ###
    def start_odas_process(self):
        k = int(self.key)
        for i in range(4):
            try:
                print(self.odas_cfgs[i])
                self.call_odas(self.odas_cfgs[i])
                self.connect_odas_audio_socket(9000+1000*i)
                self.connect_odas_sst_socket(9900+1000*i)
                logging.info('successfully started ODAS with config ' + str(i))
                return
            except Exception, e:
                print('Failed to start start ODAS with config ' + str(i))
                print(traceback.format_exc())
                self.odas_process.kill()
        print('Decoder Handler ' + str(k) + ' :No ODAS process avaiable')
        # if ODAS fails to launch on all 4 sockets, use VAD for segmentation instead
        self.odas_dead = True
        self.segmentation_method = "VAD"

    def call_odas(self, cfg):
        try:
            odas_proc = subprocess.Popen(["/opt/odas4phenopad/bin/odaslive", "-s", "-c", \
                                          "/opt/odas4phenopad/config/odaslive/respeaker2"\
                                          + "_" + cfg + '.cfg'])
            self.odas_process = odas_proc
            print("ODAS launched!")
            self.odas_pid = odas_proc.pid
            print('odas_pid', type(self.odas_pid))
        except:
            print("DecoderHandler: Cannot open ODAS")
    
    def connect_odas_audio_socket(self, port):
        try:
            self.odas_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except:
            print("error creating socket for ODAS")
        self.odas_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.odas_sock.bind(("0.0.0.0", port))  
        print("odas sock listening")
        self.odas_sock.listen(1)
        self.odas_conn, self.odas_addr = self.odas_sock.accept()
        print("\nCONNECTION:", self.odas_addr)
        #self.odas_conn = odas_conn

    def connect_odas_sst_socket(self, port):    
        try:
            self.sst_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except:
            print("error creating socket for sst")
        self.sst_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sst_sock.bind(("0.0.0.0", port))
        print("sst sock listening")
        self.sst_sock.listen(1)
        self.sst_conn, self.sst_addr = self.sst_sock.accept()
        print("\nCONNECTION:", self.sst_addr)
        #self.sst_conn = sst_conn
    
    def send_audio_to_odas(self, message):
        try:
            self.odas_conn.sendall(message)
        except:
            print("Cannot send to ODAS, changing segmentation method to VAD")
            print(traceback.format_exc())
            self.odas_dead = True
            self.segmentation_method = 'VAD'


    def recv_sst_from_odas(self):
        try:
            data = self.sst_conn.recv(4096)
            self.sst_queue.put(data)
            self.sst_trans.write(data)
        except Exception, e:
            print("cannot read from sst")
            print(traceback.format_exc())

    def close_diarization_process(self):
        try:
            if self.segmentation_method == "ODAS":
                self.odas_process.kill()
                self.odas_sock.close()
                self.sst_sock.close()
                self.sst_process.terminate()
            self.embedding_process.terminate()
        except:
            print('Warning: DecoderHandler ' + self.key + ' : closing process cannot be completed')
            print(traceback.format_exc())
    
    def restart_odas(self):
        try:
            # close current odas processes
            try:
                self.odas_process.kill()
                self.odas_sock.shutdown(socket.SHUT_WR)
                self.odas_sock.close()
                self.sst_sock.shutdown(socket.SHUT_WR)
                self.sst_sock.close()
                self.odas_conn.shutdown(socket.SHUT_WR)
                self.odas_conn.close()
                self.sst_conn.shutdown(socket.SHUT_WR)
                self.sst_conn.close()
                self.odas_addr = None
                self.sst_addr = None 
            except Exception, e:
                print('Error killing ODAS')
                print(traceback.format_exc())
            # free odas audio and sst ports
            # restart odas process
            self.start_odas_process()
            print('restart finished')
            self.restarting_odas = False
            return
        except Exception, e:
            print('Decoder ' + self.key + ' failed to restart ODAS')
            print(traceback.format_exc())
            self.restarting_odas = False
    
    ### VAD related functions ###
    def format_VAD_seg(self, seg_end):
        seg_start = self.current_seg_start
        seg_tuple = ({'start': seg_start, 'speaker': 0, 'end': seg_end, 'dir': [1.0, 0.0, 0.0]}, {'speaker': 'Speaker 1'})
        return seg_tuple

    def vad_segment_external(self, message_short):
        """ VAD segmentation for external mode. TODO: merge with vad_segment """
        self.vad_buffer.append(message_short)
        multiplier = 1
        # resize chunk to size accepted by VAD
        if len(self.vad_buffer) > 5*multiplier:
            self.frame_count += 1
            frames_list = self.vad_buffer[0:5*multiplier]
            self.vad_buffer = self.vad_buffer[5*multiplier:]
            frames_sum = ''.join(frames_list)
            frames_unpack = list(struct.unpack("%dh"%(len(frames_sum)/2), frames_sum))
            frames = []
            for i in range(16*multiplier):
                frame_unpack = frames_unpack[i*320:(i+1)*320]
                frame = struct.pack("%dh"%(320), *frame_unpack)
                frames.append(frame)
            num_speech = 0
            # process frame with VAD
            for frame in frames:
                self.num_frame += 1
                is_speech = self.vad.is_speech(frame, 16000)
                if not self.triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    if num_voiced > 0.75 * self.ring_buffer.maxlen:
                        self.triggered = True
                        if self.segmentation_method == "VAD":
                            self.current_seg_start = self.num_frame*0.02  #(320.0/16000.0) = 0.02
                            print("SPEECH SEG STARTED", self.current_seg_start)
                        self.ring_buffer.clear()
                else:
                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                    if num_unvoiced > 0.75 * self.ring_buffer.maxlen:
                        self.triggered = False
                        if self.segmentation_method == "VAD":
                            seg_end = self.num_frame*0.02
                            VAD_seg = self.format_VAD_seg(seg_end)
                            self.segmentation_result_queue.put(VAD_seg)
                            print("SPEECH SEG ENDED", seg_end)
                        self.ring_buffer.clear()
    
    def vad_segment(self, message):
        # resize chunk to size accepted by VAD
        sample_rate = 16000
        VAD_frame_duration = 0.01
        target_byte = int(sample_rate * VAD_frame_duration  * 2) # fs * dur * 2bytes/sec
        target_frame_num = int(target_byte / 2) # 2 bytes per frame

        self.vad_buffer.append(message)
        if len(self.vad_buffer) > 16:
            frames_list = self.vad_buffer[0:16]
            self.vad_buffer = self.vad_buffer[16:]
            frames_sum = ''.join(frames_list)
            frames_unpack = list(struct.unpack("%dh"%(len(frames_sum)/2), frames_sum))
            frames = []
            for i in range(16):
                frame_unpack = frames_unpack[i*target_frame_num:(i+1)*target_frame_num]
                frame = struct.pack("%dh"%(target_frame_num), *frame_unpack)
                frames.append(frame)
            # process frame with VAD
            for frame in frames:
                    self.num_frame += 1
                    is_speech = self.vad.is_speech(frame, 16000)
                    if not self.triggered:
                        self.ring_buffer.append((frame, is_speech))
                        num_voiced = len([f for f, speech in self.ring_buffer if speech])
                        if num_voiced > 0.5 * self.ring_buffer.maxlen:
                            self.triggered = True
                            if self.segmentation_method == "VAD":
                                self.current_seg_start = self.num_frame * VAD_frame_duration 
                                print("SPEECH SEG STARTED", self.current_seg_start)
                            self.ring_buffer.clear()
                    else:
                        self.ring_buffer.append((frame, is_speech))
                        num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                        if num_unvoiced > 0.5 * self.ring_buffer.maxlen:
                            self.triggered = False
                            if self.segmentation_method == "VAD":
                                seg_end = self.num_frame * VAD_frame_duration 
                                VAD_seg = self.format_VAD_seg(seg_end)
                                self.segmentation_result_queue.put(VAD_seg)
                                print("SPEECH SEG ENDED", seg_end)
                            self.ring_buffer.clear() 

    ### Diarizarion related functions ###
    def start_diarization_process(self):
        """ start a subproccess that processes SST results from ODAS
        """ 
        #TODO: rename beamforming
        self.worker.diarization_result_queue = multiprocessing.Queue()
        self.segmentation_result_queue = multiprocessing.Queue()
        
        if self.external and self.segmentation_method == "ODAS":
            # Sound source tracking / utterance segmentation using ODAS; NOTE: ODAS is not available right now 
            self.sst_trans = open("sst_trans.txt", "w")
            self.sst_process = multiprocessing.Process(target = sst_worker.start_sst, \
                                                       args = (self.sst_queue, \
                                                               self.segmentation_result_queue))
            self.sst.start()

        self.embedding_process = multiprocessing.Process(target = embedding_worker.start_embedding, \
                                                           args = (self.segmentation_result_queue, \
                                                                   self.worker.diarization_result_queue))
        self.embedding_process.start()


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)8s %(asctime)s %(message)s ")
    logging.debug('Starting up server')
    from tornado.options import define, options
    define("certfile", default="", help="certificate file for secured SSL connection")
    define("keyfile", default="", help="key file for secured SSL connection")

    tornado.options.parse_command_line()
    app = Application()
    if options.certfile and options.keyfile:
        ssl_options = {
          "certfile": options.certfile,
          "keyfile": options.keyfile,
        }
        logging.info("Using SSL for serving requests")
        app.listen(options.port, ssl_options=ssl_options)
    else:
        app.listen(options.port)
    print(type(app.status_listeners))
    print(len(app.status_listeners))
    for ws in app.status_listeners:
        print(ws)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
