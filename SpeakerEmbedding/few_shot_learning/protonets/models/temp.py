import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from protonets.models import register_model

import numpy as np

from pyannote.audio.embedding.utils import to_condensed, pdist


class TemoralAvgPooling(nn.Module):
    def __init__(self):
        super(TemoralAvgPooling, self).__init__()

    def forward(self, x):
        # x: return from LSTM
        # ((batch_size, f_samples, f_dimemsion), hidden)
        return x[0].sum(dim=1)

class UnitNormalize(nn.Module):
    def __init__(self):
        super(UnitNormalize, self).__init__()

    def forward(self, x):
        # batch_size, emb_dimension
        norm = torch.norm(x, 2, 1, keepdim=True)
        return x / norm

class FewShotSpeech(nn.Module):
    def __init__(self, encoder):
        super(FewShotSpeech, self).__init__()

        self.encoder = encoder

in_dim = 59
out_dim = 16

encoder = nn.Sequential(
        #nn.Sequential(
        nn.LSTM(in_dim, out_dim,
        bidirectional=True,
        batch_first=True),
        #),
        TemoralAvgPooling(),
        nn.Linear(out_dim * 2, out_dim, bias=True),
        nn.Tanh(),
        nn.Linear(out_dim, out_dim, bias=True),
        nn.Tanh()
        # UnitNormalize()
    )

input = Variable(torch.randn(1000, 200, 59))
h0 = Variable(torch.randn(2, 200, 59))
c0 = Variable(torch.randn(2, 200, 59))
output = encoder.forward(input)
import pdb; pdb.set_trace()