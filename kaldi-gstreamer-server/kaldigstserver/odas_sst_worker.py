""" Performs sound source tracking (sst) with ODAS to allow sound source localization 
    and utterance segmentation.
"""
import logging
import numpy as np
import math
import datetime
import webrtcvad
import traceback

import multiprocessing


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def get_angle_dif(v1, v2):
    u_v1 = unit_vector(v1)
    u_v2 = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

class Track():
    """ an utterance represented as ab ODAS sst track """
    def __init__ (self):
        # TODO
        pass

class TrackingProcess():

    def __init__ (self, odas_queue, result_queue):
        self.odas_queue = odas_queue 
        self.result_queue = result_queue
     
        self.tracks_dict = {}
        self.tracks_list = [] 

        self.speakers = ['Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4']
        self.speakers_info_dict = {}
        self.n_speaker = 3
        self.active_speaker = 0 # not used for now


    def start_tracking(self):  

    	while True: 
            sst_result_list = None
            try:
                if not self.odas_queue.empty(): 
                    sst_result_raw = self.odas_queue.get(False)
                    sst_result_list = self.extract_sst_info(sst_result_raw)
            except Exception, e:
                pass
	            #print(traceback.format_exc())
            if sst_result_list is not None:
                try: 
                    for i in range(len(sst_result_list)):
                        sst_result = self.remove_inactive_tracks(sst_result_list[i]) 

                        # active tracks tracking loop
                        for i in range(1,len(sst_result)):
                            track_id = sst_result[i][0]
                            # add track if found active track that's not recorded
                            if self.tracks_dict.get(track_id) == None and sst_result[i][3] >= 0.7:
                                self.add_track(sst_result, i)
                            elif self.tracks_dict.get(track_id) != None and sst_result[i][3] < 0.05: 		    
                                # finish track if track inactive for more than 10 frames
                                if self.tracks_dict[track_id]['inactive'] > 10:
                                    # consider track as ended after inactive for a bit
                                    finished_track = self.finish_track(sst_result, i)
                                    print("%d, %s ended: %f" % (0.032*(sst_result[0]-10), \
                                        finished_track['speaker'], self.get_angle(sst_result[i][2])))
                                    # record finished speech segment to a file
                                    self.record_finished_track(finished_track)                         
                                    # push a finished segment into result queue
                                    timespan = self.format_timespan(finished_track)
                                    apdx = {'speaker': finished_track['speaker']}
                                    if timespan['end'] - timespan['start'] > 0.3:
                                        self.result_queue.put((timespan,apdx))
                                else:
                                    # increment inactive counter if track unfinished but inactive
                                    self.tracks_dict[track_id]['inactive'] += 1

                            elif self.tracks_dict.get(track_id) != None and sst_result[i][3] >= 0.7:
                                # the track is ongoing, clear inactive counter if it's none zero, otherwise do nothing
                                if self.tracks_dict[track_id]['inactive'] > 0:
                                    self.tracks_dict[track_id]['inactive'] = 0
                            else:
                                # track is finished but haven't been terminated yet, do nothing
                                pass 
                except:
                    print(traceback.format_exc())


    def find_speaker_by_pos(self, track_status, track):
        """ assign a speaker to a track by based on the sound source location """
        n_speaker = self.n_speaker
        ssd_threshold = 0.09 
        speaker = ''
        # CASE 1: New track and no speaker registered
        if len(self.speakers_info_dict) == 0 and track_status == 'started':
            speaker = self.speakers[0]  
            return speaker
        # CASE 2: New track not all speakers registered '''
        elif len(self.speakers_info_dict) < n_speaker and track_status == 'started':
            for i in range(n_speaker):
                if self.speakers_info_dict.get(self.speakers[i]) != None:
                    ssd = self.calculate_ssd(self.speakers_info_dict[self.speakers[i]], track[2])
                    if ssd <= ssd_threshold:
                        #speaker = self.speakers.pop(i)
                        speaker = self.speakers[i]
                        return speaker
            for i in range(n_speaker):
                if self.speakers_info_dict.get(self.speakers[i]) == None:
                    #speaker = self.speakers.pop(i)
                    speaker = self.speakers[i]
                    return speaker
        # CASE 3: New track and all speakers already resgistered '''    
        elif len(self.speakers_info_dict) == n_speaker and track_status == 'started':
            min_ssd_idx = 0
            min_ssd = 12
            for i in range(n_speaker):
		        ssd = self.calculate_ssd(self.speakers_info_dict[self.speakers[i]], track[2])
		        if ssd < min_ssd:
		            min_ssd = ssd
		            min_ssd_idx = i
            speaker = self.speakers[min_ssd_idx]
            return speaker
        else:
            return speaker
        
        
    def extract_sst_info(self, sst_result): 
    	sst_split = sst_result.split("}\n{")
    	sst_info_list = []
        
        for i in range(2):
            sst_info_list.append([])
            sst_info_list[-1].append(int(sst_split[i].split("\n")[1].replace(",","").split("\"timeStamp\": ")[1]))
            for j in range(4):
                track_list = []
                vector_list = []
                vector_list.append(float(sst_split[i].split("\n")[j+3].split("\"x\": ")[1].split(",")[0]))    # build
            	vector_list.append(float(sst_split[i].split("\n")[j+3].split("\"y\": ")[1].split(",")[0]))    # vector
                vector_list.append(float(sst_split[i].split("\n")[j+3].split("\"z\": ")[1].split(",")[0]))    # list
                track_list.append(sst_split[i].split("\n")[j+3].split("\"id\": ")[1].split(",")[0])
                track_list.append(sst_split[i].split("\n")[j+3].split("\"tag\": ")[1].split(",")[0].replace("\"",""))
                track_list.append(vector_list)
                track_list.append(float(sst_split[i].split("\n")[j+3].split("\"activity\": ")[1].split(",")[0].split(" ")[0]))
                sst_info_list[-1].append(track_list)
                
            return sst_info_list
    

    def format_timespan(self, finished_track):
        timespan = {'speaker': 0, 'start': 0.0, 'end': 0.0, 'dir': []}
        timespan['speaker'] = int(finished_track['speaker'].split(' ')[1]) - 1
        timespan['start'] = round(finished_track['start'][0] * 0.032, 1)
        timespan['end'] = round(finished_track['end'][0] * 0.032, 1)
        timespan['dir'] = finished_track['end'][1]
        return timespan


    def remove_inactive_tracks(self, sst_result_in):
	    sst_result_out = [] 
	    sst_result_out.append(sst_result_in[0])
	    for i in range(1,5):
	        if sst_result_in[i][0] != '0' and sst_result_in[i][1] != '' and sst_result_in[i][2][2] < 0.99: 
		        # consider tracks with z > 0.99 as noise
		        sst_result_out.append(sst_result_in[i]) 
	    return sst_result_out


    def add_track(self, sst_result, i):
        """ This add an active track to a dictionary and initialze tracking of the track """
        offset = 10 # the start time has been experimentally found to have a ~ 0.3s lag
        track_id = sst_result[i][0]
        pos_vector = sst_result[i][2]

        self.tracks_dict[track_id] = {'start': [sst_result[0]-offset, pos_vector], \
                                              'end': [None,None], \
                                              'speaker': '', \
                                              'inactive': 0}
        self.tracks_dict[track_id]['speaker'] = self.find_speaker_by_pos('started', sst_result[i])
        if self.tracks_dict[track_id]['speaker'] != '':
            self.speakers_info_dict[self.tracks_dict[track_id]['speaker']] = pos_vector

        print(datetime.datetime.now())
        print("%d, %s started: %f" % (0.032*(sst_result[0]-10), \
                                      self.tracks_dict[track_id]['speaker'], \
                                      self.get_angle(pos_vector)))

        
    def finish_track(self, sst_result, i):
        """ This finishes tracking of the track and removes it from the dict of active tracks """
        offset = 10 # from experiments the end time has a consistent delay of ~0.3s
        track_id = sst_result[i][0]
        pos_vector = sst_result[i][2]

        self.tracks_dict[track_id]['end'] = [sst_result[0]-offset, pos_vector]
        self.tracks_dict[track_id]['id'] = track_id
        finished_track = self.tracks_dict.pop(track_id)
        if finished_track['speaker'] != '':
            self.speakers_info_dict[finished_track['speaker']] = pos_vector
        return finished_track


    def record_finished_track(self, finished_track):
        #self.tracks_list.append(finished_track)
        tracks_trans = open('/opt/test_segments.txt', 'a')
        tracks_trans.write(str(finished_track))
        tracks_trans.write('\n')
        tracks_trans.close()


    def get_audio_segment(self, start, end):
        """ Obtain a segment of audio given the timestamp at start and end """
        chunck_size = 1024
        audio_cache = open('/opt/temp_audio_file.raw', 'r')
        audio_cache.seek(0)
        audio_string = audio_cache.read()
        audio_segment = audio_string[start*chunck_size:end*chunck_size]

        return audio_segment


    def calculate_ssd(self, vec1, vec2): 
	    """ calculate the sum squred distance btw two vectors """
	    ssd = 0 
	    # normalize x and y 
	    r1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
	    x1, y1 = vec1[0]/r1, vec1[1]/r1
	    r2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
	    x2, y2 = vec2[0]/r2, vec2[1]/r2
	    ssd = (x1-x2)**2 + (y1-y2)**2

	    return ssd


    def get_angle(self, vec):
        """ calculates the horizontal angle of the position vector """
        return math.degrees(math.atan2(vec[1], vec[0]))



def start_sst(odas_queue, result_queue): 
    sst_process = TrackingProcess(odas_queue, result_queue)
    sst_process.start_tracking() 
