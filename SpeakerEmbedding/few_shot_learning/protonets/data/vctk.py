'''
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
'''


import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

from pyannote.audio.features import Precomputed
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.database import VCTK
from pyannote.core import Segment

import pickle
import random
from itertools import cycle, islice


dirname = os.path.dirname(os.path.realpath(__file__))
#VCTK_DATA_DIR = os.path.join(dirname, '../../data/vctk')
VCTK_DATA_DIR = '/w/148/spoclab/data3/jixuan/SpeakerEmbedding/few_shot_learning/data/vctk'
VCTK_AUDIO_DIR = '/p/spoclab/data3/jixuan/VCTK-Corpus/wav48'
VCTK_FEATURE_DIR = '/p/spoclab/data3/jixuan/VCTK-Corpus/playground/feature-extraction'
#VCTK_AUDIO_DIR = '/h/jixuan/Documents/data/VCTK-Corpus/wav48'
#VCTK_FEATURE_DIR = '/h/jixuan/Documents/data/VCTK-Corpus/playground/feature-extraction'
OMNIGLOT_CACHE = { }
DATASET_CACHE = { }

# precomputed = Precomputed(VCTK_FEATURE_DIR)
def get_feature(cfile, seg):
    return precomputed.crop(cfile, seg, mode='center', fixed=2.0)

def convert_tensor(key, d):
    d[key] = torch.from_numpy(np.array(d[key], np.float32, copy=False))
    return d

def convert_cuda(key, d):
    if hasattr(d[key], 'cuda'):
        d[key] = d[key].cuda()
    return d

def shuffle_dataset_index(data_ind):
    # data_ind: num_labels * num_samples_per_label
    np.random.shuffle(data_ind)
    for t in data_ind:
        np.random.shuffle(t)


'''
Sequential padding submatrix
'''
def sub_matrix(matrix, sub_w, sub_h):
    # matrix: w * h * dim_of_element

    m_h = matrix.shape[0]
    matrix = matrix[:sub_h*(m_h//sub_h), :]
    m_w = matrix.shape[1]
    matrix = matrix[:, :sub_w*(m_w//sub_w), :]

    matrix = np.array([np.split(row, m_w//sub_w,axis=1) for row in np.split(matrix, m_h//sub_h,axis=0)])
    matrix = matrix.reshape(-1, *matrix.shape[2:])
    matrix = matrix.reshape(matrix.shape[0], -1, matrix.shape[-1])
    return matrix

def batch_from_index(dataset, b_index):
    # dataset: num_labels * num_data
    # b_index: num_samples * 2 (data_index, label_index)
    batch_feature = []
    batch_labels = []
    for data_idx, label_idx in b_index:
        seg, cfile = dataset[label_idx][data_idx]
        batch_feature.append(get_feature(cfile, seg))
        batch_labels.append(label_idx)
    return {
        'data': batch_feature,
        'class': batch_labels
    }

'''
Loading dataset sequentially after shuffling
Make sure we go through every data in each epoch
'''
def load(opt, splits):
    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        # duration = opt['data.duration']
        ret[split] = VCTKLoader(split=split, n_support=n_support, n_query=n_query, n_way=n_way, if_cuda=opt['data.cuda'])
    return ret



class VCTKLoader:
    def __init__(self, split='', n_support=0, n_query=0, n_way=0, if_cuda=False):
        self.precomputed = Precomputed(VCTK_FEATURE_DIR)
        self.dataset = self.load_dataset(from_disk=True)

        self.split = split
        self.n_support = n_support
        self.n_query = n_query
        self.n_way = n_way
        self.dataset = None
        self.transforms = None
        self.if_cuda = if_cuda


    def shuffle_dataset(self):
        num_label = len(self.dataset['class'])
        num_data = len(self.dataset['data'][0]) # x: num_labels * num_samples_per_label

        # num_label * num_data
        data_index = np.tile(np.arange(num_data), (num_label,1))
        label_index = np.tile(np.arange(num_label).reshape(-1, 1), (1,num_data))
        label_index = np.expand_dims(label_index, axis=2)
        data_index = np.expand_dims(data_index, axis=2)
        data_label_idx = np.concatenate((data_index, label_index), axis=2)

        # shuffle rows (labels)
        np.random.shuffle(data_label_idx)
        # shuffle data index for each row (for each label)
        for dl in data_label_idx:
            np.random.shuffle(dl)

        index_batches = sub_matrix(data_label_idx, self.n_support+self.n_query, self.n_way)
        # optional: shuffle batches
        np.random.shuffle(index_batches)
        return index_batches

    def __iter__(self):
        if self.dataset is None:
            self.dataset = self.load_dataset(from_disk=True)[self.split]
            transforms = [partial(batch_from_index, self.dataset['data']), partial(convert_tensor, 'data')]
            if self.if_cuda:
                transforms.append(CudaTransform())
            self.transforms = compose(transforms)
        index_batches = self.shuffle_dataset()
        batches = TransformDataset(ListDataset(index_batches), self.transforms)

        print(f"\nSize of batches: {len(batches)}")
        for batch in batches:
            batch['n_way'] = self.n_way
            batch['n_support'] = self.n_support
            batch['n_query'] = self.n_query
            yield batch


    def get_feature(self, cfile, seg):
        return self.precomputed.crop(cfile, seg, mode='center', fixed=2.0)
        # return precomputed(cfile).crop(seg, mode='center', fixed=2.0)

    def load_speaker_file(self, protocol_name='SpeakerEmbedding.All', from_disk=False):
        database = VCTK()
        protocol = database.get_protocol(protocol_name.split('.')[0], protocol_name.split('.')[1])

        speaker_file = { 'train': {}, 'val': {}, 'test':{}, 'unseen':{}}
        if from_disk:
            print('Loading speaker_file from disk...')
            vctk_file_name = os.path.join(VCTK_DATA_DIR, 'vctk_speaker_file')
            if not os.path.isfile(vctk_file_name):
                raise ValueError(f'{vctk_file_name} not found')
            else:
                with open(vctk_file_name, 'rb') as vctk_file:
                    speaker_file = pickle.load(vctk_file)
                return speaker_file

        print('Loading unseen set...')
        for current_file in protocol.unseen_iter():
            speaker = current_file['uri'].split('_')[0]
            if not speaker in speaker_file['unseen']:
                speaker_file['unseen'][speaker] = []
            speaker_file['unseen'][speaker].append(current_file)

        print('Loading training set...')
        for current_file in protocol.train():
            speaker = current_file['uri'].split('_')[0]
            if not speaker in speaker_file['train']:
                speaker_file['train'][speaker] = []
            speaker_file['train'][speaker].append(current_file)

        print('Loading test set...')
        for current_file in protocol.test():
            speaker = current_file['uri'].split('_')[0]
            if not speaker in speaker_file['test']:
                speaker_file['test'][speaker] = []
            speaker_file['test'][speaker].append(current_file)

        print('Loading development set...')
        for current_file in protocol.development():
            speaker = current_file['uri'].split('_')[0]
            if not speaker in speaker_file['val']:
                speaker_file['val'][speaker] = []
            speaker_file['val'][speaker].append(current_file)


        with open(os.path.join(VCTK_DATA_DIR, 'vctk_speaker_file'), 'wb') as vctk_file:
            pickle.dump(speaker_file, vctk_file,  -1)
        return speaker_file


    def load_speaker_segments(self, seg_dur=2.0, overlap_ratio=0.25, from_disk=False):
        '''
        2 seconds segments, with overlapping ratio = 0.25
        |----||----|
           |----|
        '''
        spk_seg = { 'train': {}, 'val': {}, 'test':{}, 'unseen':{}}
        if from_disk:
            print('Loading speaker_segments from disk...')
            vctk_file_name = os.path.join(VCTK_DATA_DIR, 'vctk_speaker_segments')
            if os.path.isfile(vctk_file_name):
                with open(vctk_file_name, 'rb') as vctk_file:
                    spk_seg = pickle.load(vctk_file)
                return spk_seg
            else:
                raise ValueError(f'{vctk_file_name} not found')

        def fetch_spk_seg(speaker_file):
            speaker_seg = {}
            for spk, sfiles in speaker_file.items():
                speaker_seg[spk] = []

                for sfile in sfiles:
                    duration = sfile['annotated'].duration()
                    if duration < seg_dur:
                        continue
                    half_seg = seg_dur / 2
                    for mid in np.arange(half_seg, duration-half_seg, seg_dur*(1-overlap_ratio)):
                        speaker_seg[spk].append(
                            (Segment(mid-half_seg, mid+half_seg), sfile)
                        )
            return speaker_seg

        spk_file = self.load_speaker_file(from_disk=from_disk)
        for sub in ['train', 'val', 'test', 'unseen']:
            spk_file[sub] = fetch_spk_seg(spk_file[sub])

        with open(os.path.join(VCTK_DATA_DIR, 'vctk_speaker_segments'), 'wb') as vctk_file:
            pickle.dump(spk_file, vctk_file,  -1)
        return spk_file

    def load_dataset(self, from_disk=False):
        print('Loading dataset...')
        dataset = { 'train': {}, 'val': {}, 'test':{}, 'unseen':{}}
        if from_disk:
            vctk_file_name = os.path.join(VCTK_DATA_DIR, 'vctk_datasets')
            if os.path.isfile(vctk_file_name):
                with open(vctk_file_name, 'rb') as vctk_file:
                    dataset = pickle.load(vctk_file)
                    return  dataset
            else:
                raise ValueError(f'{vctk_file_name} not found')

        spk_seg = self.load_speaker_segments(from_disk=from_disk)
        with open(os.path.join(VCTK_DATA_DIR, 'vctk_datasets'), 'wb') as vctk_file:
            for sub in dataset.keys():
                speaker_segments = spk_seg[sub]
                seg_count = []
                for spk, seg in speaker_segments.items():
                    seg_count.append(len(seg))
                max_count = max(seg_count)
                for spk in speaker_segments.keys():
                    speaker_segments[spk] = list(islice(cycle(speaker_segments[spk]), max_count))
                y_labels = speaker_segments.keys()
                dataset[sub] = {
                    'class': list(y_labels),
                    'data': [speaker_segments[label] for label in y_labels]
                }
            pickle.dump(dataset, vctk_file,  -1)
        return dataset

    # load MFCC features into memory
    # take too much memory, not recommand
    def load_features(self, from_disk=False):
        feature_dataset = {'train': {}, 'val': {}, 'test': {}, 'unseen':{}}
        if from_disk:
            vctk_file_name = os.path.join(VCTK_DATA_DIR, 'vctk_feature_datasets')
            if os.path.isfile(vctk_file_name):
                with open(vctk_file_name, 'rb') as vctk_file:
                    feature_dataset = pickle.load(vctk_file)
                    return feature_dataset
            else:
                raise ValueError(f'{vctk_file_name} not found')

        dataset = self.load_dataset(from_disk=True)
        with open(os.path.join(VCTK_DATA_DIR, 'vctk_feature_datasets'), 'wb') as vctk_file:
            for sub in feature_dataset.items():
                print(f'Loading feature: {sub}')
                subset = dataset[sub]
                for spk, seg_list in subset.items():
                    print(f'Speaker: {spk}')
                    feature_list = []
                    for seg, cfile in seg_list:
                        feature_list.append(self.get_feature(cfile, seg))
                    feature_dataset[sub][spk] = np.array(feature_list)
            pickle.dump(feature_dataset, vctk_file, -1)
        return feature_dataset

    # load data for same/different experiments
    def load_same_diff_data(self, from_disk=False):
        exp_dataset = {'train': {},'val': {}, 'test': {}, 'unseen': {}}
        n_pair = 40
        n_pair_unseen = 100
        print(f'Loading same/diff data, #pair: {n_pair}, #pair_unseen: {n_pair_unseen}')
        file_name = os.path.join(VCTK_DATA_DIR, f'same_diff_exp_norepeat_{n_pair}_{n_pair_unseen}')

        if from_disk:
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as dfile:
                    exp_dataset = pickle.load(dfile)
                    return exp_dataset
            else:
                raise ValueError(f'{file_name} not found, generate first?')

        def gen_same_diff(data, labels, first_n=-1, n_same_pair=20):
            same_pairs = []
            diff_pairs = []
            for spk, spk_data in zip(labels, data):
                first_n = len(spk_data) # if first_n == -1 or first_n > len(spk_data) else first_n
                # get same pairs
                ind = list(range(first_n))
                n = 0
                same_pair_ind = []
                ind_his = set()
                while n < n_same_pair:
                    i1 = random.choice(ind)
                    ind.remove(i1)
                    i2 = random.choice(ind)
                    if (not (i1, i2) in ind_his) and (not (i2, i1) in ind_his):
                        ind_his.add((i1, i2))
                        same_pair_ind.append((i1, i2))
                        n += 1
                    else:
                        print('skip repeated pair')
                    ind = list(range(first_n))

                spk_same_pairs = [(spk_data[ind[0]], spk_data[ind[1]]) for ind in same_pair_ind]
                same_pairs.extend(spk_same_pairs)

            # get different pairs
            labels_ind = list(range(len(labels)))
            n = 0
            pair_his = set()
            while n < n_same_pair * len(labels):
                s1 = random.choice(labels_ind)
                ind1 = random.choice(list(range(len(data[s1]))))
                labels_ind.remove(s1)
                s2 = random.choice(labels_ind)
                ind2 = random.choice(list(range(len(data[s2]))))

                pair1 = (s1, ind1); pair2 = (s2, ind2)
                if (not (pair1, pair2) in pair_his ) and (not (pair2, pair1) in pair_his):
                    pair_his.add((pair1, pair2))
                    diff_pairs.append((data[s1][ind1], data[s2][ind2]))
                    n += 1
                else:
                    print('skip repeated pair')
                labels_ind = list(range(len(labels)))
            print(len(same_pairs), len(diff_pairs))
            return same_pairs, diff_pairs

        print("Loading dataset from disk, instead of generating from scratch")
        dataset = self.load_dataset(from_disk=True)
        for subset in ['train', 'val', 'test', 'unseen']:
            first_n = -1 #  if subset == 'unseen' else 200
            npair = n_pair_unseen if subset == 'unseen' else n_pair
            data = dataset[subset]['data']
            labels = dataset[subset]['class']
            same_pairs, diff_pairs = gen_same_diff(data, labels, first_n=first_n, n_same_pair=npair)
            assert len(same_pairs) == len(diff_pairs)
            assert len(same_pairs) == len(labels) * npair
            exp_dataset[subset] = {
                'same': same_pairs,
                'diff': diff_pairs
            }
        with open(os.path.join(VCTK_DATA_DIR, file_name), 'wb') as dfile:
            pickle.dump(exp_dataset, dfile, -1)
        return exp_dataset




if __name__ == '__main__':
    vctk = VCTKLoader(split='train', n_support=5, n_query=5, n_way=15, if_cuda=True)
    e = vctk.load_same_diff_data(False)
    import pdb; pdb.set_trace()



#### backup code for randomly generating batches
"""
'''
Batch generator using pyannote, randomly generate batches
For each epoch, ensure seeing one miniute audio data for each speaker
'''
def randomly_load(opt, splits):
        #train
        print("Initializing batch generator for validation")
        batch_generator.initialize(protocol, subset='development')
        batches = batch_generator(protocol, subset='development')
        # train
        print("Initializing batch generator for training")
        batch_generator.initialize(protocol, subset='train')
        batches = batch_generator(protocol, subset='train')

        _ = {
            'batches': transformed_batches,
            'batches_per_epoch': getattr(batch_generator, 'batches_per_epoch', None)
        }

def get_batch_generator(per_class, n_way, duration):

    audio_features = Precomputed(root_dir=VCTK_FEATURE_DIR)# , use_memmap=False)

    # label_min_duration: used to remove speakers with little audio data, arbitrarily chosen 40 seconds
    return SpeechSegmentGenerator(
        audio_features, label_min_duration=2.0 * 20,
        per_label=per_class, per_fold=n_way,
        duration=duration, min_duration=None,
        max_duration=None, parallel=1)
def transform_batches(batches, use_cuda):
    for batch in batches:
        '''
            batch = {
                'X': batch_size * n_sample * n_feature,
                'y': label index,
                'y_database': database index,
                'extra': {
                    'label': label list,
                    'database': database list
                }
            }
        '''
        batch = convert_tensor('X', batch)

        if use_cuda:
            batch = convert_cuda('X', batch)
        yield batch

def transform_batches_few_shot(batches, n_way, n_support, n_query, use_cuda):
    for batch in batches:
        '''
            batch = {
                'X': batch_size * n_sample * n_feature,
                'y': label index,
                'y_database': database index,
                'extra': {
                    'label': label list,
                    'database': database list
                }
            }
        '''
        batch = convert_tensor('X', batch)
        n_example = n_support + n_query
        batch['X'] = batch['X'].view(n_way, n_example,
                                        *batch['X'].size()[1:])
        batch['xs'] = batch['X'][:,:n_support,:,:]
        batch['xq'] = batch['X'][:,n_support:,:,:]

        if use_cuda:
            batch = convert_cuda('X', batch)
            batch = convert_cuda('xs', batch)
            batch = convert_cuda('xq', batch)
        yield batch

def sequential_batch(dataset, x_index, y_index, n_way, per_class):
    num_y = len(y_index) # num_labels
    num_x = len(x_index[0]) # x: num_labels * num_samples_per_label
    for i in range(num_y // n_way):
        for j in range(num_x // per_class):
            y_ind = y_index[i*n_way:(i+1)*n_way]
            x_ind = x_index[y_ind, :][:, j*per_class:(j+1)*per_class]
            batch = []
            for m in range(n_way):
                temp = []
                for n in range(per_class):
                    yi = y_ind[m]
                    xi = x_ind[m, n]
                    (seg, cfile) = dataset[yi][xi]
                    temp.append(get_feature(cfile, seg))
                batch.append(np.array(temp))
            yield np.swapaxes(np.array(batch), 0, 1)
"""
