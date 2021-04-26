import torch
import yaml
import os.path
import numpy as np
import functools
from docopt import docopt
import pickle

from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier
from pyannote.database import get_protocol

from pyannote.audio.features.utils import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.utils import PyannoteFeatureExtractionError


def load_mfcc_feature_extractor(config_yml):
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))
    return feature_extraction


def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class SequencialEpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way):
        self.n_classes = n_classes
        self.n_way = n_way

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        perm_classes = torch.randperm(self.n_classes)
        for i in range(0, self.n_classes-self.n_way, self.n_way):
            yield perm_classes[i : i + self.n_way]

class MatrixSequentialBatchSampler(object):
    def __init__(self, num_label, num_data, n_way, per_class):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])