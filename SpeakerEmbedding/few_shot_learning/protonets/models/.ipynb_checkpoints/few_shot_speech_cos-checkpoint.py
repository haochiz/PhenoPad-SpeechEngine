import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from protonets.models import register_model
#from .utils import euclidean_dist
from .utils import cosine_dist

import numpy as np

from pyannote.audio.embedding.utils import to_condensed, pdist

class TristouNetEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TristouNetEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rnn = nn.LSTM(in_dim, out_dim,
            bidirectional=True,
            batch_first=True)
        self.remaining = nn.Sequential(
            TemoralAvgPooling(),
            nn.Linear(out_dim * 2, out_dim, bias=True),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.Tanh(),
            UnitNormalize()
        )
        self.num_directions_ = 2

    def forward(self, x):
        pass
        '''
        batch_size, _, n_features = x.size()
        h = torch.zeros(self.num_directions_, batch_size, self.out_dim,
                                requires_grad=False)
        c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                        device=device, requires_grad=False)
        hidden = (h, c) 
        output, _ = layer(x, hidden)
        return self.remaining.forward(output)
        '''


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

    def pdist(self, fX):
        """Compute pdist à-la scipy.spatial.distance.pdist

        Parameters
        ----------
        fX : (n, d) torch.Tensor
            Embeddings.

        Returns
        -------
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix
        """

        n_sequences, _ = fX.size()
        distances = []

        for i in range(n_sequences - 1):
            d = 1. - F.cosine_similarity(
                fX[i, :].expand(n_sequences - 1 - i, -1),
                fX[i+1:, :], dim=1, eps=1e-8)

            distances.append(d)

        return torch.cat(distances)

    def generate_triplet(self, y, distances):
        """Build all possible triplet

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives

    def triplet_loss(self, distances, anchors, positives, negatives):
        """Compute triplet loss

        Parameters
        ----------
        distances : torch.Tensor
            Condensed matrix of pairwise distances.
        anchors, positives, negatives : list of int
            Triplets indices.
        return_delta : bool, optional
            Return delta before clamping.

        Returns
        -------
        loss : torch.Tensor
            Triplet loss.
        """
        # estimate total number of embeddings from pdist shape
        n = int(.5 * (1 + np.sqrt(1 + 8 * len(distances))))
        n = [n] * len(anchors)

        # convert indices from squared matrix
        # to condensed matrix referential
        pos = list(map(to_condensed, n, anchors, positives))
        neg = list(map(to_condensed, n, anchors, negatives))

        # compute raw triplet loss (no margin, no clamping)
        # the lower, the better
        delta = distances[pos] - distances[neg]

        # clamp triplet loss
        #TODO Add to command params
        margin_ = 0.2
        loss = torch.clamp(delta + margin_, min=0)

        # return triplet losses
        return loss, delta.view((-1, 1)), pos, neg

    def loss(self, batch):
        n_class = batch['n_way']
        n_support = batch['n_support']
        n_query = batch['n_query']
        assert batch['data'].size(0) == n_class * (n_support + n_query)

        xs = batch['data'].view(n_class, n_support+n_query, *batch['data'].size()[1:])[:, :n_support, :] # support
        xq = batch['data'].view(n_class, n_support+n_query, *batch['data'].size()[1:])[:, n_support:, :] # query 

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.contiguous().view(n_class * n_support, *xs.size()[2:]), 
                        xq.contiguous().view(n_class * n_query, *xq.size()[2:])],
                        0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        # dists = euclidean_dist(zq, z_proto)
        dists = cosine_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('few_shot_speech_cos')
def load_few_short_speech(**kwargs):
    in_dim = kwargs['in_dim'][-1]
    out_dim = kwargs['out_dim']
    
    # encoder = TristouNetEncoder(in_dim, out_dim)
    encoder = nn.Sequential(
            nn.Sequential(
                nn.LSTM(in_dim, out_dim,
                bidirectional=True,
                batch_first=True)
            ),
            TemoralAvgPooling(),
            nn.Linear(out_dim * 2, out_dim, bias=True),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.Tanh(),
            UnitNormalize()
        )

    return FewShotSpeech(encoder)
