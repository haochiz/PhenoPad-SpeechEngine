""" Under construction.
    Considerations: -> reliably terminate odas process when speech
                       session stops
                    -> restarting odas process if it crashes
                    -> managing multiple parallel odas processes
                       for multiple users (future)    
"""
import time
import traceback 
import socket
import subprocess

EXIT_MSG = "EXIT"

class Beamforming():
    def __init__ (self, audio_queue, sst_queue):
        self.auido_queue = audio_queue
        self.sst_queue = sst_queue
        # init socket related variables
        self.odas_sock = None
        self.odas_conn = None
        self.sst_sock = None
        self.sst_conn = None
        self.odas_pid = None
        self.odas_process = None
        self.odas_cfgs = ['a', 'b', 'c', 'd']
        # start odas
        #self.start_odas_process()
        # start beamforming process
        #self.beamforming_process()



    def call_odas(self, cfg):
        try:
            odas_proc = subprocess.Popen(["/opt/odas4phenopad/bin/odaslive", "-s", "-c", \
                                          "/opt/odas4phenopad/config/odaslive/respeaker2"\
                                          + "_" + cfg + '.cfg'])
            self.odas_process = odas_proc
            print("ODAS launched!")
            self.odas_pid = odas_proc.pid
            print('odas pid', self.odas_pid)
            # TODO: send odas_proc.pid to master_server
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

    def start_odas_process(self):
        for i in range(4):
            try:
                print(self.odas_cfgs[i])
                self.call_odas(self.odas_cfgs[i])
                self.connect_odas_audio_socket(9000+1000*i)
                self.connect_odas_sst_socket(9900+1000*i)
                #logging.info('successfully started ODAS with config ' + str(i))
                return
            except Exception, e:
                print('Failed to start start ODAS with config ' + str(i))
                print(traceback.format_exc())
                self.odas_process.kill()
        print('Beamforming Worker ' + ' :Error creating ODAS process')

    def send_audio_to_odas(self, message):
        try:
            self.odas_conn.sendall(message)
        except:
            print('Cannot send to ODAS')
            print(traceback.format_exc())
            #TODO: send error msg to master_server

    def recv_sst_from_odas(self):
        try:
            data = self.sst_conn.recv(2048)
            self.sst_queue.put(data, block=True)
            print '111'
        except Exception, e:
            print("cannot read from sst")
            print(traceback.format_exc())

    def close_odas_process(self):
        try:
            self.odas_process.kill()
            self.odas_sock.close()
            self.sst_sock.close()
        except:
            print('DecoderHandler ' + self.key + ' : failed to close ODAS processes')
            print(traceback.format_exc()) 
    

    def beamforming_process(self):
        self.sst_queue.put(self.odas_pid, block=True)
        while True:
            print 'i am alive'
            try:
                message = None
                try:
                    if not self.auido_queue.empty():
                        message = self.auido_queue.get(False)
                except Exception, e:
                    print(traceback.format_exc())
                if message:
                    if message == EXIT_MSG:
                        print('Beamforming Worker: On Close')
                        self.close_odas_process()
                        return
                    else:
                        self.send_audio_to_odas(message)
                self.recv_sst_from_odas()
            except Exception, e:
                print(traceback.format_exc())

                    


def start_beamforming_process(audio_queue, sst_queue):
    beamforming = Beamforming(audio_queue, sst_queue)
    beamforming.start_odas_process()
    beamforming.beamforming_process()

