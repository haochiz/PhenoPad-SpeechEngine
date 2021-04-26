import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import squareform
from protonets.models import register_model
from .utils import euclidean_dist
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

class TristouNet(nn.Module):
    def __init__(self, encoder):
        super(TristouNet, self).__init__()

        self.encoder = encoder

    def pdist(self, fX):
        """Compute pdist -la scipy.spatial.distance.pdist

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
            
            # d = torch.pow(fX[i, :].expand(n_sequences - 1 - i, -1) - fX[i+1:, :], 2).sum(1)
            distances.append(d)

        return torch.cat(distances)

    def generate_triplet_hards(self, y, distances):
        """Build triplet with both hardest positive and hardest negative
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

        distances = squareform(self.to_numpy(distances))
        y = np.array(y)

        for anchor, y_anchor in enumerate(y):

            d = distances[anchor]

            # hardest positive
            pos = np.where(y == y_anchor)[0]
            pos = [p for p in pos if p != anchor]
            positive = int(pos[np.argmax(d[pos])])

            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives

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

    def generate_triplet_hard_randomly(self, y, distances):
        """Build triplet with hardest negative
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

        distances = squareform(distances)
        y = np.array(y)
    

        for anchor, y_anchor in enumerate(y):

            # hardest negative
            d = distances[anchor]
            neg = np.where(y != y_anchor)[0]
            np.random.shuffle(neg)
            # negative = int(neg[np.argmin(d[neg])])

            for positive in np.where(y == y_anchor)[0]:
                if positive == anchor:
                    continue

                for negative in neg:
                    # select one hard example
                    if d[positive] > d[negative] - 0.2:
                        anchors.append(anchor)
                        positives.append(positive)
                        negatives.append(negative)
                        break

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
        # batch: n_sample_batch * f_samples * f_dimemsion
        # n_sample_batch = n_class * n_seg
        '''
        batch['X'] = torch.tensor(np.stack(batch['X']),
                                    dtype = torch.float32,
                                    device=device)
        '''
        n_way = batch['n_way']
        n_support = batch['n_support']
        n_query = batch['n_query']
        assert batch['data'].size(0) == n_way * (n_support + n_query)
                

        # z: n_sample_batch * z_dim
        z = self.encoder.forward(batch['data'])
        y = batch['class']
    
        # pre-compute pairwise distances
        distances = self.pdist(z)

        # sample triplets
        # anchors, positives, negatives = self.generate_triplet(y, distances)
        anchors, positives, negatives = self.generate_triplet_hard_randomly(y, distances.data.cpu().numpy())

        # compute loss for each triplet
        losses, delta, _, _ = self.triplet_loss(distances, anchors, positives, negatives)

        loss_val = losses.mean()

        # accuracy
        z_support = z.view(n_way, n_support+n_query, -1)[:, :n_support, :]
        z_support_mean = z_support.mean(1) # size: n_way * z_dim
        z_query = z.view(n_way, n_support+n_query, -1)[:, n_support:, :]
        z_query = z_query.contiguous().view(n_way * n_query, -1)
        target_inds = torch.arange(0, n_way).view(n_way, 1).expand(n_way, n_query).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if z.is_cuda:
            target_inds = target_inds.cuda()
        
        dists = cosine_dist(z_query, z_support_mean) # size: (n_way*n_query) * n_way
        pred_inds = dists.argmin(1)
        pred_inds = pred_inds.view(n_way, n_query)
        acc_val = torch.eq(pred_inds, target_inds).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
        
    def evaluate(self, batch):
        # batch: n_sample_btch * f_samples * f_dimemsion
        # n_sample_batch = n_class * n_seg
        '''
        batch['X'] = torch.tensor(np.stack(batch['X']),
                                    dtype = torch.float32,
                                    device=device)
        '''
        n_way = batch['n_way']
        n_support = batch['n_support']
        n_query = batch['n_query']
        assert batch['data'].size(0) == n_way * (n_support + n_query)
                

        # z: n_sample_batch * z_dim
        z = self.encoder.forward(batch['data'])
        y = batch['class']
    
        loss_val = -1

        # accuracy
        z_support = z.view(n_way, n_support+n_query, -1)[:, :n_support, :]
        z_support_mean = z_support.mean(1) # size: n_way * z_dim
        z_query = z.view(n_way, n_support+n_query, -1)[:, n_support:, :]
        z_query = z_query.contiguous().view(n_way * n_query, -1)
        target_inds = torch.arange(0, n_way).view(n_way, 1).expand(n_way, n_query).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if z.is_cuda:
            target_inds = target_inds.cuda()
        
        dists = cosine_dist(z_query, z_support_mean) # size: (n_way*n_query) * n_way
        pred_inds = dists.argmin(1)
        pred_inds = pred_inds.view(n_way, n_query)
        acc_val = torch.eq(pred_inds, target_inds).float().mean()

        return loss_val, {
            'loss': -1,
            'acc': acc_val.item()
        }

@register_model('tristou_net')
def load_tristounet(**kwargs):
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

    return TristouNet(encoder)
