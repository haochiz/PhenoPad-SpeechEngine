import torch 
import numpy
import random
import yaml
import pickle
import struct
from scipy import spatial
import scipy
from scipy.ndimage import gaussian_filter

import traceback

import webrtcvad
from feature_extraction.yaafe import YaafeMFCC
from feature_extraction.feature import SlidingWindowFeature

BYTES_PER_FRAME = 2
SAMPLING_RATE = 16000

USE_SPEAKER_ENBEDDING = True

MIN_DIST = 4.0 # recording the globally min euclidean dist, used to normalize dist

def load_seg(mfcc):
    spk_seg = []
    for i in range(0, (mfcc.shape[0]-200)//100+1):
        spk_seg.append(mfcc[i*100:i*100+200, :])
        #print numpy.array(spk_seg).shape
    return spk_seg

def inter_dist(spk_emb):
    spk_inter = []
    for i in range(spk_emb.shape[0]):
        for j in range(i+1, spk_emb.shape[0]):
            spk_inter.append(numpy.linalg.norm(spk_emb[i]-spk_emb[j]))
    return spk_inter

def intra_dist(spk1_emb, spk2_emb):
    spk_intra = []
    for i in range(spk1_emb.shape[0]):
        for j in range(0, spk2_emb.shape[0]):
            spk_intra.append(numpy.linalg.norm(spk1_emb[i]-spk2_emb[j]))
    return spk_intra

def manh_dist(spk1_emb, spk2_emb):
    spk_intra = []
    for i in range(spk1_emb.shape[0]):
        for j in range(0, spk2_emb.shape[0]):
            spk_intra.append(spatial.distance.cityblock(spk1_emb[i],spk2_emb[j]))
    return spk_intra

def new_dist(spk1_emb, spk2_emb):
    # experiment: try Lp norm where p < 1
    p = 0.5
    dist_ = 0
    for i in range(spk1_emb.shape[1]):
        dist_ += (abs(spk1_emb[0][i] - spk2_emb[0][i]))**p
    dist = dist_**(1.0/p)
    return dist 


def diarization_chunk_to_nice_length(feature_extraction, all_bytes):
    block_size = feature_extraction.block_size
    step_size = feature_extraction.step_size

    left_num = (len(all_bytes) // BYTES_PER_FRAME)  % step_size
    num_block = len(all_bytes) // BYTES_PER_FRAME  // step_size
    if left_num < block_size - step_size:
        """ not enough bytes for last block """
        num_block -= 1
        left_num = step_size + left_num

    desire_sample_num = num_block * step_size + block_size - step_size

    desire_bytes = all_bytes[:desire_sample_num * BYTES_PER_FRAME]
    leftover_bytes = all_bytes[-left_num * BYTES_PER_FRAME:]
    return desire_bytes, leftover_bytes


class Speaker():
    """ speaker object, each represent one speaker detected by diarization agent
    """
    def __init__(self, idx, name, embedding, direction_vec=numpy.array([])):
        self.name = name
        self.index = idx
        if direction_vec.size > 0:
            self.direction_vec = direction_vec # numpy array
        else:
            self.direction_vec = numpy.array([])
        self.embedding = embedding # a 2x1 array that contains the embedding vector and it's weight
        self.embedding_list = [embedding] # a list keeping track of all embeddings assigned to the speaker

    def add_embedding(self, embedding, direction_vec=numpy.array([])):
        self.embedding_list.append(embedding)
        # update embedding vector
        # emb_vec = (current_emb_vec*current_weight + new_emb_vec*new_emb_weight)/sum_weight
        self.embedding[0] = (self.embedding[0]*self.embedding[1] + embedding[0]*embedding[1])/(self.embedding[1]+embedding[1])
        self.embedding[1] = self.embedding[1] + embedding[1]
        # update direction_vec
        if self.direction_vec.size > 0 and direction_vec.size > 0:
            self.direction_vec = 0.4*direction_vec + (1-0.4)*self.direction_vec
        elif self.direction_vec.size <= 0 and direction_vec.size > 0:
            self.direction_vec = direction_vec
            

class Utterance():
    """ utterance object, each represent one utterance
        defined as a class in case some logic will be added in the future
    """
    def __init__(self, start, end, speaker, embedding, direction_vec=numpy.array([])):
        self.start = start
        self.end = end
        self.speaker = speaker # speaker idx of speaker assigned to the utterance
        self.embedding = embedding
        if direction_vec.size > 0:
            self.direction_vec = direction_vec
        else:
            self.direction_vec = numpy.array([])

class Cluster():
    """ a class used to keep track of clusters during offline clustering;
        each object is a cluster resulted from an offline clustering
    """
    def __init__(self, init_point):
        # init_point is the first data point (an utterance object) assign to the cluster
        self.embedding_centroid = init_point.embedding
        self.points = [init_point] # a list that contains all data points

    def add_point(self, point):
        # add a data point to the cluster
        self.points.append(point)
        # update centroid
        self.embedding_centroid[0] = (self.embedding_centroid[0]*self.embedding_centroid[1] + \
                                      point.embedding[0]*point.embedding[1])/(self.embedding_centroid[1]+point.embedding[1])
        self.embedding_centroid[1] = self.embedding_centroid[1] + point.embedding[1]        

class Kmeans():
    """ class for keeping track of kmeans results;
        each object is a result of one kmeans clustering
    """
    def __init__(self, K, clusters):
        # K is the num of clusters, and clusters is the list of the k resulted clusters 
        self.K = K
        self.clusters = clusters # each cluster in clusters is an cluster object

    def compute_SSE(self):
        SSE = 0
        for i in range(len(self.clusters)):
            SSE_ = 0
            for j in range(len(self.clusters[i].points)):
                SSE_ += (intra_dist(self.clusters[i].points[j].embedding[0], self.clusters[i].embedding_centroid[0])[0])**2
            SSE += SSE_
        return SSE 


class SpeakerEmbedding(): 

    def __init__ (self, segments_queue, embedding_queue):
        self.segments_queue = segments_queue
        self.embedding_queue = embedding_queue

        self.speaker_assignment_method = "CRP" # simple/CRP

        self.n_speaker = 5
        self.speaker_names = ['Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4', 'Speaker 5', 'Speaker 6'] # names to be given to potential speakers
        self.speakers_dict = {} # {speaker_idx: speaker_obj}
        self.direction_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5} # this might be outdated
        #self.segments_list = [] # stores segments that are shorter than 2s, isn't used atm, TODO: think about what to do with this
        #self.embeddings_list  = [] # stores all embedding results TODO: decide if this is still necessary
        self.short_utterance_list = []
        self.long_utterance_list = []
        self.recluster_count = 0

        self.vad = webrtcvad.Vad(3) # Vad aggressiveness is 3 

        config_yml =  '/home/haochi/speechengine_test/kaldi-gstreamer-server/kaldigstserver/model/config_mfcc_25_10.yml'       
        with open(config_yml, 'r') as fp:
            self.config = yaml.load(fp)
            FeatureExtraction = YaafeMFCC
            self.feature_extraction = FeatureExtraction(**self.config['feature_extraction'].get('params', {}))
            fp.close()

        self.model = torch.load('/home/haochi/speechengine_test/SpeakerEmbedding/few_shot_learning/pretrained_models/best_model_3.pt')
        self.model.eval()

        self.audio_file = '/home/haochi/speechengine_test/temp_audio_file.raw'    

    def embedding_process(self):
        # embedding process loop
        print('embedding process started')
        while True:
            segment = None 
            try:
                segment = self.segments_queue.get(False)
                print(segment)
            except Exception, e:
                # Queue Empty exception
                ##print(traceback.format_exc())
                pass        
            try:
                if segment:
                    embedding_result = self.get_embedding_result(segment)
                    # 1st item of embedding_result indicate if embedding can be successfully calculated
                    if embedding_result[0]: 
                        # assign speaker
                        if self.speaker_assignment_method == "simple":
                            # NOTE: this method can only be used with ODAS. TODO: maybe update this so it can not only work with ODAS
                            # use ODAS to initialize if num embedding smaller than threshold
                            if len(self.long_utterance_list) < 6.0: 
                                # TODO: change arguments
                                speaker_result = self.assign_speaker(embedding_result[1], segment[0]['speaker'], segment[0]['dir'])                         
                            else:
                                speaker_result = self.assign_speaker(embedding_result[1], None, segment[0]['dir'])
                            utterance = Utterance(segment[0]['start'], segment[0]['end'], \
                                                  speaker_result.index, embedding_result[1], \
                                                  direction_vec=numpy.array(segment[0]['dir']))
                            self.long_utterance_list.append(utterance)
                            segment[0]['speaker'] = speaker_result.index
                        elif self.speaker_assignment_method == "CRP":
                            speaker_result = self.assign_speaker_CRP(embedding_result[1], segment[0]['dir'])
                            utterance = Utterance(segment[0]['start'], segment[0]['end'], \
                                                  speaker_result.index, embedding_result[1], \
                                                  direction_vec=numpy.array(segment[0]['dir']))                          
                            self.long_utterance_list.append(utterance)
                            segment[0]['speaker'] = speaker_result.index

                        # update direction_map, only useful when using ODAS
                        # TODO: change the default speaker value for single channel so this doesn't apply
                        self.direction_map[segment[0]['speaker']] = speaker_result.index
                        result_seg = self.format_result_seg(segment[0])
                        self.embedding_queue.put((result_seg, True))
                        print('Speaker Embedding Worker: put: ', segment[0])
                        print(len(self.long_utterance_list))
                    else:
                        # if utterance shorter than 2s, return (False, default speaker)
                        segment[0]['speaker'] = self.direction_map[segment[0]['speaker']]
                        utterance = Utterance(segment[0]['start'], segment[0]['end'], \
                                              segment[0]['speaker'], embedding_result[1], \
                                              direction_vec=numpy.array(segment[0]['dir']))  
                        self.short_utterance_list.append(utterance)
                        result_seg = self.format_result_seg(segment[0])
                        self.embedding_queue.put((result_seg, True))
                        print('put: ', segment[0])

                # recluster every n segments
                if len(self.long_utterance_list) >= (self.recluster_count+1)*10: # recluster after every x data point
                    print('reclustering')
                    self.recluster_count += 1
                    recluster_result = self.recluster()
                    temp_utterance_list = []
                    self.speakers_dict = {}
                    # update Speakers objects with new clusters
                    for i,c in enumerate(recluster_result.clusters):
                        speaker = Speaker(i, self.speaker_names[i], c.points[0].embedding)
                        self.speakers_dict[i] = speaker
                        c.points[0].speaker = i
                        temp_utterance_list.append(c.points[0])
                        for j in range(1, len(c.points)):
                            self.speakers_dict[i].add_embedding(c.points[j].embedding)
                            c.points[j].speaker = i
                            temp_utterance_list.append(c.points[j])
                    
                    # reorder utterance here
                    start_time_list = []
                    self.long_utterance_list = []
                    for u in temp_utterance_list:
                        start_time_list.append(u.start)
                    # add utterances in temp_list to long_utterance_list
                    # by the order of earliest start time to lastest
                    while len(temp_utterance_list) > 0:
                        min_idx = start_time_list.index(min(start_time_list))
                        self.long_utterance_list.append(temp_utterance_list.pop(min_idx))
                        start_time_list.pop(min_idx)

                    # testing affinity matrix method
                    """
                    embedding_vecs = []
                    for u in self.long_utterance_list:
                        embedding_vecs.append(u.embedding[0])
                        
                    print("affinity best k", find_num_cluster_affinity(embedding_vecs))
                    """
                    # push result to queue
                    recluster_result_list = []
                    for u in self.long_utterance_list:
                        recluster_result_list.append({'start': u.start, \
                                                      'end': u.end, \
                                                      'speaker': u.speaker})
                        print({'start': u.start, 'end': u.end, 'speaker': u.speaker})
                    self.embedding_queue.put((recluster_result_list, False))
                    print('put recluster result')
                    
            except Exception, e:
                print(traceback.format_exc())


    def get_embedding_result(self, segment):
        """ calculate the embedding of a given segment, returns (True, embedding) if segment longer than 2s
            else returns (False, defualt_speaker )
        """
        if segment[0]['end'] - segment[0]['start'] >= 2.0:
            full_audio_data = self.get_audio_data()
            # reshape audio data
            audio_data = numpy.array(full_audio_data, dtype=numpy.float64, order='C').reshape(1, -1)
            # convert to mfcc
            features = self.feature_extraction.extract_from_bytes(audio_data)
            # extract features for segment
            segment_features = features.data[int(round(segment[0]['start']*100)):int(round(segment[0]['end']*100)), :]
            segment_segments = load_seg(segment_features)
            
            rnn_output, _ = self.model.encoder_rnn.forward(torch.Tensor(numpy.array(segment_segments)))
            seq_len = torch.ones(rnn_output.size(0)) * rnn_output.size(1)
            z = self.model.encoder_linear.forward((rnn_output, seq_len))
            segment_embedding = z.data.numpy()
            segment_embedding = numpy.mean(segment_embedding, axis = 0, keepdims=True)

            return (True, [segment_embedding, len(segment_segments)]) 

        else:
            # segment not long enough
            return (False, segment[0]['speaker'])


    def get_audio_data(self):
        input_file = open(self.audio_file, 'rb')
        input_bytes = input_file.read()
        desire_bytes, leftover_bytes = diarization_chunk_to_nice_length(self.feature_extraction, input_bytes)
        raw_audio_data = numpy.frombuffer(desire_bytes, dtype=numpy.int16)
        return raw_audio_data


    def assign_speaker(self, embedding_result, sst_speaker, dir_vec):
        DIRECTION_THRESHOLD = 6.0  # how many segments to use only beamforming result
        EMBEDDING_THRESHOLD = 2.5

        embedding_vec = embedding_result[0]
        embedding_weight = embedding_result[1]
        dir_vec = numpy.array(dir_vec)
        # if no speaker has been assigned
        if len(self.speakers_dict) == 0:
            # assign new speaker as the sst_speaker
            print('CASE 1')
            speaker_idx = sst_speaker 
            speaker_name = self.speaker_names[speaker_idx]
            speaker_embedding = [embedding_vec, embedding_weight]
            speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
            self.speakers_dict[speaker_idx] = speaker

            return speaker
        # if some available speakers assigned but not all
        elif len(self.speakers_dict) < self.n_speaker:
            # first see if embedding close enough to any speaker's embedding
            print('CASE 2')
            min_dist = 100.0
            min_idx = None
            sum_dist = 0
            for i in range(len(self.speakers_dict)):
                # calculate dist to each speaker's embedding in speakers_dict 
                if self.speakers_dict.get(i):
                    dist = intra_dist(self.speakers_dict[i].embedding[0], embedding_vec)[0]
                    sum_dist = sum_dist + dist
                    print (i, dist)
                    if dist < EMBEDDING_THRESHOLD and dist < min_dist: 
                        min_idx = i
                        min_dist = dist
            if min_idx != None:
                if len(self.long_utterance_list) >= DIRECTION_THRESHOLD: # preset threshould
                    # assigning to existing speaker
                    speaker_idx = min_idx               
                    print('using embedding')
                    speaker = self.speakers_dict[speaker_idx]
                    speaker_embedding = [embedding_vec, embedding_weight]
                    speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)
                else: 
                    # assign to ODAS speaker
                    speaker_idx = sst_speaker
                    if self.speakers_dict.get(speaker_idx):
                        # assign to existing speaker
                        speaker = self.speakers_dict[speaker_idx]
                        speaker_embedding = [embedding_vec, embedding_weight]
                        speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)
                    else:
                        # assign to new speaker
                        speaker_name = self.speaker_names[speaker_idx]
                        speaker_embedding = [embedding_vec, embedding_weight]
                        speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
                        self.speakers_dict[speaker_idx] = speaker

                return speaker

            else:
                if len(self.long_utterance_list) >= DIRECTION_THRESHOLD:
                    # assign to new speaker
                    print("using embedding")
                    speaker_idx = len(self.speakers_dict)
                    speaker_name = self.speaker_names[speaker_idx]
                    speaker_embedding = [embedding_vec, embedding_weight]
                    speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
                    self.speakers_dict[speaker_idx] = speaker
                else:
                    # assign to ODAS speaker
                    speaker_idx = sst_speaker
                    if self.speakers_dict.get(speaker_idx):
                        # assign to existing speaker
                        speaker = self.speakers_dict[speaker_idx]
                        speaker_embedding = [embedding_vec, embedding_weight]
                        speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)                     
                    else:
                        # assign to new speaker
                        speaker_name = self.speaker_names[speaker_idx]
                        speaker_embedding = [embedding_vec, embedding_weight]
                        speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
                        self.speakers_dict[speaker_idx] = speaker                   
                        self.speakers_dict[speaker] = [embedding_vec, embedding_weight]

                return speaker

        # if all available speakers assigned, assigned the one closest to embedding
        elif len(self.speakers_dict) == self.n_speaker:
            print('CASE 3')
            min_dist = 100.0
            min_idx = None
            for i in range(len(self.speakers_dict)):
                if self.speakers_dict.get(i):
                    dist = intra_dist(self.speakers_dict[i].embedding[0], embedding_vec)[0]
                    print(i,dist)
                    if dist < min_dist:
                        min_idx = i
                        min_dist = dist

            if min_idx != None:
                if len(self.long_utterance_list) >= DIRECTION_THRESHOLD: # preset threshould
                    # assign to existing speaker
                    print('using embedding')
                    speaker_idx = min_idx
                    speaker = self.speakers_dict[speaker_idx]
                    speaker_embedding = [embedding_vec, embedding_weight]
                    speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)                 
                else:
                    # assign to ODAS speaker
                    speaker_idx = sst_speaker
                    speaker = self.speakers[speaker_idx]
                    speaker_embedding = [embedding_vec, embedding_weight]
                    speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)                 

                return speaker

    def assign_speaker_CRP(self, embedding_result, dir_vec):
        ''' use exp(-2d/a) instead;  a = 1 for now
        '''
        # TODO: organize code here
        a = 0.8
        alpha = 0.027
        embedding_vec = embedding_result[0]
        embedding_weight = embedding_result[1] 
        dir_vec = numpy.array(dir_vec)

        global MIN_DIST

        # if no speaker has been assigned
        if len(self.speakers_dict) == 0: 
            speaker_idx = 0
            speaker_name = self.speaker_names[speaker_idx]
            speaker_embedding = [embedding_vec, embedding_weight]
            # add new speaker
            speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
            self.speakers_dict[speaker_idx] = speaker           

            return speaker

        # disable alpha if reached max num of speakers
        if len(self.speakers_dict) >= self.n_speaker:
            alpha = 0.0

        # compute distance to all current centroids 
        distances = {}
        for i in range(len(self.speakers_dict)):
            # calculate cos dist of direction vec
            if dir_vec.size > 0 and self.speakers_dict[i].direction_vec.size > 0:
                cos_dis = spatial.distance.cosine(dir_vec, self.speakers_dict[i].direction_vec)
            else: 
                cos_dis = 0.0
            print('cos dis', cos_dis)
            # Euclidean Distance
            euc_dis = intra_dist(self.speakers_dict[i].embedding[0], embedding_vec)[0]
            distances[i] = euc_dis + 0.5*cos_dis
            if euc_dis < MIN_DIST: MIN_DIST = euc_dis
            # Manhattan Distance
            '''
            manh_distance = manh_dist(self.speakers_dict[i].embedding[0], embedding_vec)
            distances[i] = manh_distance[0]/4.5 + 0.5*cos_dis
            '''
            # Lp p = 0.5
            '''
            test_distance = new_dist(self.speakers_dict[i].embedding[0], embedding_vec)
            distances[i] = test_distance/119.51 + 0.5*cos_dis
            '''
        # compute probability for each cluster assignment
        fod, probs = {}, {} # fod: f(d)
        sqr_sum = alpha**2
        # compute distance function
        for key in distances: 
            fod[key] = numpy.exp(-distances[key]/a)
            sqr_sum += (fod[key])**2
        # compute confidence scores (probability)
        for key in distances: probs[key] = fod[key]**2/sqr_sum
        probs['alpha'] = alpha**2/sqr_sum 

        # print values for debug purposes
        print('distances: ', distances)
        print('probs: ', probs)
        print('adjusted probs', probs)

        # assign speaker and update centroid/new cluster
        r = random.random()
        bound = [0.0, 0.0]
        # if the highest confidence score >= x, assign key as speaker directly
        if max(probs.values()) >= 0.5:
            key = probs.keys()[probs.values().index(max(probs.values()))]
            if key == 'alpha':
                # assign to new speaker
                speaker_idx = len(self.speakers_dict)
                speaker_name = self.speaker_names[speaker_idx]
                speaker_embedding = [embedding_vec, embedding_weight]
                speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
                self.speakers_dict[speaker_idx] = speaker                   
                print('r: %s; speaker: %s'%(r, speaker.name))
                return speaker
            else:
                # assign to existing speaker
                speaker_idx = key
                speaker = self.speakers_dict[speaker_idx]
                speaker_embedding = [embedding_vec, embedding_weight]
                speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)
                print('r: %s; speaker: %s'%(r, speaker.name))
                return speaker
        # else assign randomly according to probabality
        for key in probs:
            bound[1] += probs[key]
            if r >= bound[0] and r < bound[1]:
                if key == 'alpha': 
                    # assign to new speaker 
                    speaker_idx = len(self.speakers_dict)
                    speaker_name = self.speaker_names[speaker_idx]
                    speaker_embedding = [embedding_vec, embedding_weight]
                    speaker = Speaker(speaker_idx, speaker_name, speaker_embedding, direction_vec=dir_vec)
                    self.speakers_dict[speaker_idx] = speaker                   
                    print('r: %s; speaker: %s'%(r, speaker.name))
                    return speaker                  
                else:
                    # assign to existing speaker
                    speaker_idx = key
                    speaker = self.speakers_dict[speaker_idx]
                    speaker_embedding = [embedding_vec, embedding_weight]
                    speaker.add_embedding(speaker_embedding, direction_vec=dir_vec)
                    print('r: %s; speaker: %s'%(r, speaker.name))
                    return speaker                  
            else:
                bound[0] += probs[key] 


    def adjust_probs(self, probs):
        """ This function is unused, original purpose is to adjust probs according to 
            num of points in each cluster
        """
        N_probs = {}
        weights = {} 
        alpha = 10 #?
        max_prob = max(probs.values())
        # construct N_probs
        N_probs['alpha'] = 10
        # NOTE: self.speakers_dict[key][2] kept tracks of how many seg (data points) are assigned to the 
        #       speaker, now this number can simply be obtained from len(speaker.embedding_list)
        for key in self.speakers_dict: N_probs[key] = numpy.log(self.speakers_dict[key][2]+1)
        # construct weights 
        for key in probs: weights[key] = numpy.exp(-2.5*(max_prob - probs[key]))
        # compute N_probs
        sum_N_probs = sum(N_probs.values())
        for key in N_probs: N_probs[key] = N_probs[key]/sum_N_probs
        # adjust probs
        for key in probs: probs[key] = probs[key] + weights[key]*N_probs[key]
        print('probs1', probs)
        sum_probs = sum(probs.values())
        for key in probs: probs[key] = probs[key]/sum_probs
        print('probs2', probs)

        return probs

    # TODO: rewrite recluster methods
    def recluster(self):
        # Method 1: Kmeans++ and elbow
        # perform kmeans++ on for 1 to n number of clusters
        """
        kmeans_results = [] # list of Kmeans objects
        for k in range(1, self.n_speaker+1):
            temp_results = []
            temp_SSEs = []
            # kmeans++ for each k 3 times, use the result with the best SSE
            for i in range(10):
                temp_results.append(kmeans_plusplus(k, self.long_utterance_list))
                temp_SSEs.append(temp_results[-1].compute_SSE())
            min_idx = temp_SSEs.index(min(temp_SSEs))
            kmeans_results.append(temp_results[min_idx])

        # performing elbow method on kmeans results 
        num_cluster = find_num_cluster_elbow(kmeans_results)
        # TODO: decide if should redo kmeans or used previous results
        # using previous result with num_cluster of clusters for now
        recluster_result = kmeans_results[num_cluster-1]
        """ 
        # Method 2: Affinity and Kmeans++
        # compute best k with affinity matrix
        embedding_vecs = []
        for u in self.long_utterance_list:
            embedding_vecs.append(u.embedding[0])

        num_cluster = find_num_cluster_affinity(embedding_vecs, self.n_speaker)
        print("affinity best k", num_cluster)

        # cluster with kmeans++ for k clusters
        temp_results = []
        temp_SSEs = []
        for i in range(10):
            temp_results.append(kmeans_plusplus(num_cluster, self.long_utterance_list))
            temp_SSEs.append(temp_results[-1].compute_SSE())
        min_idx = temp_SSEs.index(min(temp_SSEs))
        recluster_result = temp_results[min_idx]
        
        return recluster_result
                
    def find_closest_cluster(self, embedding_tup):
        # TODO: integrate dir_vec
        embedding = embedding_tup[0]
        weight = embedding_tup[1]
        min_dist = 100.0
        min_cluster = None
        for key in self.speakers_dict:
            dist = intra_dist(self.speakers_dict[key][0], embedding)
            if dist[0] < min_dist:
                min_dist = dist[0]
                min_cluster = key
        return min_cluster

    # function for BIC/xmeans, not used for now
    def merge_clusters(self, clusters_dict): 
        """ Check pairs of clusters that are close to each other and merge the 
            ones that results in higher BIC when merged
        """
        temp_dict = {}
        len_dict = {}
        key_list = []

        for key in clusters_dict:
            key_list.append(key)
            temp_dict[key] = []
            len_dict[key] = 0
            #len_dict[key] = len(clusters_dict[key])
            for i in range(len(clusters_dict[key])):
                temp_dict[key].append(clusters_dict[key][i][0][0])
                len_dict[key] += clusters_dict[key][i][1]
            print(numpy.array(temp_dict[key]).shape)

        should_merge = self.compare_BIC_for_pairs(numpy.array(temp_dict[key_list[0]]), \
                                                  numpy.array(temp_dict[key_list[1]]), \
                                                  len_dict[key_list[0]], len_dict[key_list[1]])   

        return clusters_dict 

    # BIC calculation doesn't seem to return correct result, at least on small samples
    def compare_BIC_for_pairs(self, cluster1, cluster2, n1, n2):
        ''' TESTING: function unfinished
        '''
        #print('cluster1', cluster1)
        n3 = n1 + n2
        merge = numpy.concatenate((cluster1, cluster2))

        #print('n1 n2 n3 N', n1, n2, n3, N)
        #N = len(self.embeddings_list)
        N = 0
        d = 32
        for key in self.speakers_dict: N += self.speakers_dict[key][1]

        #alpha = N/2000           # penalty for not having enough sample points
        beta = max(10.0**(300-30*self.recluster_count), 1.0)  # scaling factor to avoid round very small det to zero

        cov_mtx1 = numpy.cov(cluster1.T) 
        cov_mtx2 = numpy.cov(cluster2.T)
        cov_mtx3 = numpy.cov(merge.T)
        cov_mtx1[0,:] = cov_mtx1[0,:]*beta
        cov_mtx2[0,:] = cov_mtx2[0,:]*beta
        cov_mtx3[0,:] = cov_mtx3[0,:]*beta
        #d = cov_mtx3.shape[0]
        print('n1 n2 n3 N', n1, n2, n3, N)
        print('cov_mtx shapes', cov_mtx1.shape, cov_mtx2.shape)
        print('det: %s, %s, %s' % (numpy.linalg.det(cov_mtx3), numpy.linalg.det(cov_mtx1), numpy.linalg.det(cov_mtx2)))
        delta = (-1*n3*(numpy.log(abs(numpy.linalg.det(cov_mtx3)))-numpy.log(beta))) - \
            ((-1*n1*(numpy.log(abs(numpy.linalg.det(cov_mtx1)))-numpy.log(beta))) + \
             (-1*n2*(numpy.log(abs(numpy.linalg.det(cov_mtx2)))-numpy.log(beta)))) + \
            0#N*(d + 0.5*d*(d + 1))
        print('delta', delta)

        return False


    def format_result_seg(self, segment):
        result_seg = {}
        result_seg['start'] = abs(segment['start'])
        result_seg['end'] = abs(segment['end'])
        result_seg['speaker'] = abs(segment['speaker'])         

        return result_seg        

    # this function is only used in if_speech (which is no longer used) so is also no used
    def format4VAD(self, audio_segment):
        """ repack audio segment into 320 bytes frames requriedby VAD
        """
        frames = []
        while len(audio_segment) > 5*1024*2:
            #frames_list = audio_segment[0:5]
            #audio_segment = audio_segment[5:]
            #frames_sum = ''.join(frames_list)
            frames_sum = audio_segment[0:5*1024*2]
            audio_segment = audio_segment[5*1024*2:]
            frames_unpack = list(struct.unpack("%dh"%(len(frames_sum)/2), frames_sum))
            for i in range(16):
                frame_unpack = frames_unpack[i*320:(i+1)*320]
                frame = struct.pack("%dh"%(320), *frame_unpack)
                frames.append(frame)

        return frames

    # this function is no loneer necessary so is not used
    def if_speech(self, segment):
        """ determine if an audio segment contains enough active speech
        """
        # TODO: extract audio segment from segment info
        start = int(segment[0]['start']/0.0000625) # idx of the starting byte
        end = int(segment[0]['end']/0.0000625)     # idx of the ending byte 
        input_file = open(self.audio_file, 'rb')
        input_bytes = input_file.read()
        input_data = list(struct.unpack("%dh"%(len(input_bytes)/2), input_bytes))
        audio_seg_list = input_data[start:end]
        audio_seg = struct.pack("%dh"%(len(audio_seg_list)), *audio_seg_list)
        # TODO: repackage audio seg to frames with correct sizes
        frames = self.format4VAD(audio_seg)
        # TODO: determine if the audio seg has high enough ratio of active speech
        num_speech = 0
        for frame in frames:
            if self.vad.is_speech(frame, 16000):
                num_speech += 1
        if float(num_speech)/float(len(frames)) >= 0.3: # ratio beyond which the segment is considered active
            print('ratio', float(num_speech)/float(len(frames)))
            return True    
        else:
            return False


def kmeans_plusplus(k, points):
    """ this function performs kmeans++ clustering;
        k is the desired num of clusters,
        points is a list that contains all the data points,
        each item in points is an utterance object
    """
    # initialize first cluster
    p0 = points[0]
    cluster = Cluster(p0)
    clusters = [cluster]
    centroid_idxs = [0] # keeps tracks of the idx of points being assigned as centroids
    # initialize of the rest of the clusters
    for i in range(1, k):
        # compute the squared distances from each point to the closest cluster
        squared_dists = scipy.array([min([intra_dist(p.embedding[0], c.embedding_centroid[0])[0] for c in clusters]) for p in points])
        probs = probs = squared_dists/squared_dists.sum()
        cumprobs = numpy.cumsum(probs)
        r = scipy.rand()
        for j, prob in enumerate(cumprobs):
            if r < prob:
                idx = j
                break
        # initialize cluster
        cluster = Cluster(points[idx])
        clusters.append(cluster)
        centroid_idxs.append(idx)
    # assign points to clusters
    for i, p in enumerate(points):
        if i not in centroid_idxs:
            # find closest cluster to the point
            min_dist = 1000.0
            min_idx = 0
            for j, c in enumerate(clusters):
                if intra_dist(p.embedding[0], c.embedding_centroid[0])[0] < min_dist:
                    min_idx = j
                    min_dist = intra_dist(p.embedding[0], c.embedding_centroid[0])[0]
            # add point to closest cluster
            clusters[min_idx].add_point(p)
    # save result as a Kmean object
    kmean_result = Kmeans(k, clusters)

    return kmean_result

def find_num_cluster_elbow(kmeans_results):
    """ elbow method for estimating appropriate num of clusters;
        kmeans_results is the list of different Kmeans object with different k
    """
    threshold = 3.0 # threshold for deteremining k
    # get SSE for each result with different k
    SSE_list = []
    for i, km in enumerate(kmeans_results):
        SSE_list.append(km.compute_SSE())
    # get the rate of change of SSEs
    ratios = [] # rate of change of SSEs
    '''
    for i in range(len(SSE_list)-1):
        RoCs.append(abs(SSE_list[i] - SSE_list[i+1]))
    print(RoCs)
    '''
    print(SSE_list)
    # determine best k
    ##best_k = len(kmeans_results)
    best_k = 1
    for i in range(len(SSE_list)-2):
        ratio = max(0.0001, (SSE_list[i]-SSE_list[i+1]))/max(0.0001, (SSE_list[i+1]-SSE_list[i+2]))
        ratios.append(ratio)
        if abs(SSE_list[i]-SSE_list[i+1]) < 1.0:
            '''
            best_k = i + 1
            print("best_k", best_k)
            return best_k
            '''
            pass

    for i,r in enumerate(ratios):
        if r >= threshold:
            best_k = i + 2
            print("best_k", best_k)
            return best_k
        ''' 
        elif ratio <= 0.3333333:
            best_k = i + 2
            break
        '''
    print("best_k", best_k)
    return best_k

def find_num_cluster_affinity(vectors, max_k):
    """ estimate the number of cluster using the affinity matrix
        vectors is the list of embedding vectors of all utterances
    """
    # affinity matrix calculation
    affinity = affinity_matrix(vectors)
    # post process affinity
    pp_affinity = post_process_affinity(affinity)
    eigen_values, eigen_vecs = numpy.linalg.eig(pp_affinity)
    eigen_values = numpy.real(eigen_values)
    for i,n in enumerate(eigen_values):
        if n < 0.01:
            eigen_values[i] = 0.0

    # sort the eigenvalues from largest to smallest and find the k largest values
    eigen_values = numpy.sort(eigen_values)
    eigen_values = numpy.flip(eigen_values)
    print(eigen_values)
    ratios = []
    for i in range(len(eigen_values)-1):
        r = eigen_values[i]/eigen_values[i+1]
        if numpy.isnan(r) or numpy.isinf(r):
            r = 0.0
        ratios.append(r)
    ratios = ratios[:max_k] # make sure number of clusters does not exceed n speaker
    k = ratios.index(max(ratios)) + 1
    print(ratios) 
    return k
    
def affinity_matrix(vecs):
    """ build affinity matrix 
        vecs is a matrix represented by a list of vectors
        returns the affinity matrix (numpy matrix)
    """
    global MIN_DIST
    A = numpy.empty([len(vecs),len(vecs)])
    ##print(A)
    '''
    for i in range(len(vecs)): 
        ##print(i)
        for j in range(len(vecs)):
            if numpy.isnan(numpy.corrcoef(vecs[i], vecs[j])[0][1]): 
                A[i,j] = 0.0
            else:   
                A[i,j] = numpy.corrcoef(vecs[i], vecs[j])[0][1]
    '''
    # use normalized euclidean distance
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            ##A[i,j] = (4.0 - intra_dist(vecs[i], vecs[j])[0]) / 4.0
            A[i,j] = 1.0 / ((intra_dist(vecs[i], vecs[j])[0] - MIN_DIST) + 1.0) 
    
    MIN_DIST = 4.0

    return A
                
def post_process_affinity(matrix):
    """ post process affinity matrix
        matrix is a numpy matrix which is the affinity matrix
        returns the post processed numpy matrix 
    """
    A = matrix
    # Gaussian filter
    A_blur = gaussian_filter(A, sigma=1.0)
    # Row-wise threshould
    for row in A_blur: 
        qpercentile = numpy.percentile(row, 60)
        for i in range(len(row)): 
            if row[i] < qpercentile: 
                row[i] = 0.01 * row[i]

    # Symmetrization
    for i in range(len(A_blur)):
        ##print(i)
        for j in range(len(A_blur)):
            if A_blur[i,j] < A_blur[j,i]:
                A_blur[i,j] = A_blur[j,i]
            else:
                A_blur[j,i] = A_blur[i,j]
                
    # Diffusion
    Y = numpy.matmul(A_blur, A_blur.T)  
    
    # Normalization 
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i,j] = Y[i,j]/numpy.ndarray.max(Y[i])
            
    return Y


def start_embedding(segments_queue, embedding_queue):
    embedding_process = SpeakerEmbedding(segments_queue, embedding_queue)
    embedding_process.embedding_process()

