
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
from protonets.data.base import convert_dict, CudaTransform, \
                                SequencialEpisodicBatchSampler, load_mfcc_feature_extractor, \
                                EpisodicBatchSampler, SequentialBatchSampler

from pyannote.audio.features import Precomputed
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.core import Segment

from pyannote.audio.features.utils import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.utils import PyannoteFeatureExtractionError

import pickle
import random
from itertools import cycle, islice
import glob
import logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh) 

dirname = os.path.dirname(os.path.realpath(__file__))

#VOXCELEB_MAIN_DIR =  '/p/spoclab/data3/jixuan/VoxCeleb2/'
#VOXCELEB_DATA_DIR = VOXCELEB_MAIN_DIR + 'playground'

LIBRISPEECH_DATA_DIR = '/h/jixuan/gobi2/librispeech_features/mfcc_25_10'


def load_dataset():
    logger.info('Loading dataset')
    with open(os.path.join(LIBRISPEECH_DATA_DIR, 'librispeech_dataset.pkl'), 'rb') as dfile:
        dataset = pickle.load(dfile)
    return dataset
        
dataset = load_dataset() 

config_yml = '/h/jixuan/gobi2/vox2_features/config/config_mfcc_25_10.yml'
feature_extraction = load_mfcc_feature_extractor(config_yml)

def get_mfcc_by_audio(speaker, afile):
    base = os.path.basename(afile)
    # dirname = os.path.dirname(afile)
    fname = os.path.splitext(base)[0]
    return os.path.join(MFCC_DIR, speaker, f'{fname}.mfcc')
    
    
def extract_audio_mfcc(speaker, key, out_field, d):
    mfcc = get_mfcc_by_audio(speaker, d['key'])
    if not os.path.isfile(mfcc):
        raise Exception(f"Cannot find MFCC file: {mfcc}")
        
    with open(mfcc, 'rb') as dfile:
        d[out_field] = pickle.load(dfile)
    return d

def convert_tensor(keys, d):
    for key in keys:
        d[key] = torch.from_numpy(np.array(d[key], np.float32, copy=False))
    return d

def load_class_audio(split, d):
    class_audio = dataset[split][d['class']]

    if len(class_audio) == 0:
        raise Exception(f"No audio found for speaker {d['class']}")

    audio_ds = TransformDataset(ListDataset(class_audio),
                                compose([partial(convert_dict, 'file_name'),
                                         partial(extract_audio_mfcc, d['class'], 'file_name', 'data')]))

    loader = torch.utils.data.DataLoader(audio_ds, batch_size=len(audio_ds), shuffle=False)
    
    for sample in loader:
        data = sample
        break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': data }

def rand_mfcc(mfcc_f, min_len, max_len):
    with open(mfcc_f, 'rb') as df:
        mfcc = pickle.load(df)
        r_len = np.random.randint(min_len, max_len)
        if mfcc.shape[0] <= r_len:
            return mfcc
        else:
            start_ind = np.random.randint(0, mfcc.shape[0]-r_len)
            return mfcc[start_ind : start_ind+r_len]      

'''
extract episode (or batch)
'''
def extract_episode(key, dataset, min_len, max_len, n_support, n_query, d):
    speaker = d['class']
    mfcc_files = dataset[speaker]
    rand_files = np.random.choice(mfcc_files, n_support+n_query)
    support_files = rand_files[:n_support]
    query_files = rand_files[n_support:]
                        
    xs = []
    xs_len = []
    xs_padded = []
    for file in support_files:
        mfcc_ = rand_mfcc(file, min_len, max_len)
        pad_len = max_len - mfcc_.shape[0]
        padded_mfcc = np.pad(mfcc_, ((0,pad_len),(0,0)), 'constant', constant_values=(0))
        xs.append(mfcc_)
        xs_padded.append(padded_mfcc)
        xs_len.append(mfcc_.shape[0])
    xs = np.asarray(xs)
    xs_padded = np.asarray(xs_padded)
    xs_len = np.asarray(xs_len)
            
    xq = []
    xq_len = []
    xq_padded = []
    for file in query_files:
        mfcc_ = rand_mfcc(file, min_len, max_len)
        pad_len = max_len - mfcc_.shape[0]
        padded_mfcc = np.pad(mfcc_, ((0,pad_len),(0,0)), 'constant', constant_values=(0))
        xq.append(mfcc_)
        xq_padded.append(padded_mfcc)
        xq_len.append(mfcc_.shape[0])
    xq = np.asarray(xq)
    xq_padded = np.asarray(xq_padded)
    xq_len = np.asarray(xq_len)
                    
    return {
        'class': d['class'],
        'xs_padded': xs_padded,
        'xq_padded': xq_padded,
        'xs_len': xs_len,
        'xq_len': xq_len
    }

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

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']
        
        speaker_ids = dataset[split]['class']
        data_split = dataset[split]['data']
                     
        transforms = [partial(convert_dict, 'class'),
                      partial(extract_episode, 'class', data_split,
                              opt['data.min_len'], opt['data.max_len'],
                              n_support, n_query),
                     partial(convert_tensor,
                             ['xq_padded', 'xs_padded', 'xq_len', 'xs_len'])]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)
        
        ds = TransformDataset(ListDataset(speaker_ids), transforms)
        
        #sampler = SequencialEpisodicBatchSampler(len(ds), n_way)
        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret

if __name__ == '__main__':
    fake_opt = {}
    fake_opt['data.way'] = 10
    fake_opt['data.shot'] = 5
    fake_opt['data.query'] = 5
    fake_opt['data.test_way'] = 10
    fake_opt['data.test_shot'] = 5
    fake_opt['data.test_query'] = 5
    fake_opt['data.min_len'] = 100
    fake_opt['data.max_len'] = 500
    fake_opt['data.cuda'] = True
    
    epoch = load(fake_opt, ['train', 'val'])
    for batch in epoch['val']:
        print(batch)
        import pdb; pdb.set_trace()
                    