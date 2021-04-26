'''
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
'''


import os
import sys
import glob
import yaml
import functools

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
from pyannote.core import Segment

from pyannote.audio.features.utils import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.utils import PyannoteFeatureExtractionError

import pickle
import random
from itertools import cycle, islice


dirname = os.path.dirname(os.path.realpath(__file__))

#VOXCELEB_MAIN_DIR =  '/p/spoclab/data3/jixuan/VoxCeleb2/'
#VOXCELEB_DATA_DIR = VOXCELEB_MAIN_DIR + 'playground'

VOXCELEB_DATA_DIR = '/h/jixuan/gobi2/vox2_features'


def get_feature(segment):
    '''
    seg = {
        'file': os.path.join(dirname, fname+'.mfcc'),
        'seg': (ind, ind+seg_dim)
    }
    '''
    with open(segment['file'], 'rb') as f:
        mfcc = pickle.load(f)
        return mfcc[segment['seg'][0] : segment['seg'][1]]

    raise ValueError(f'MFCC file {segment["file"]} not found')

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
        seg = dataset[label_idx][data_idx]
        batch_feature.append(get_feature(seg))
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
        ret[split] = VoxCelebLoader(split=split, n_support=n_support, n_query=n_query, n_way=n_way, if_cuda=opt['data.cuda'])
    return ret



class VoxCelebLoader:
    def __init__(self, split='', n_support=0, n_query=0, n_way=0, if_cuda=False):
        self.dataset = self.load_dataset(which='vox2_dev_dataset.pkl', from_disk=True)

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


    def get_feature(self, segment):
        '''
        seg = {
            'file': os.path.join(dirname, fname+'.mfcc'),
            'seg': (ind, ind+seg_dim)
        }
        '''
        with open(segment['file'], 'rb') as f:
            mfcc = pickle.load(f)
            return mfcc[segment['seg'][0] : segment['seg'][1]]

        raise ValueError(f'MFCC file {segment["file"]} not found')


    def load_dataset(self, which='voxceleb_dataset', from_disk=True):
        print(f'Loading dataset {which}...')
        if from_disk:
            fname = os.path.join(VOXCELEB_DATA_DIR, which)
            if os.path.isfile(fname):
                with open(fname, 'rb') as vf:
                    dataset = pickle.load(vf)
                    return  dataset
            else:
                raise ValueError(f'{fname} not found')

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
    print(':)')
