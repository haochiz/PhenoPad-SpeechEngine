import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from protonets.models import register_model
from .utils import euclidean_dist

import numpy as np

from pyannote.audio.embedding.utils import to_condensed, pdist

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

class TemoralAvgPooling(nn.Module):
    def __init__(self):
        super(TemoralAvgPooling, self).__init__()

    def forward(self, x):
        # x[0] unpadded sequences, x[1] lengths
        # here, we are using pack_padded_sequence, .data is needed for PackedSequence
        return x[0].data.sum(dim=1) / x[1].unsqueeze(1)
        # return x[0].data.sum(dim=1)

class UnitNormalize(nn.Module):
    def __init__(self):
        super(UnitNormalize, self).__init__()

    def forward(self, x):
        # batch_size, emb_dimension
        norm = torch.norm(x, 2, 1, keepdim=True)
        return x / norm

class FewShotSpeech(nn.Module):
    def __init__(self, encoder_rnn, encoder_linear):
        super(FewShotSpeech, self).__init__()

        self.encoder_rnn = encoder_rnn
        self.encoder_linear = encoder_linear

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

    
    # training with various length segmets 
    def loss(self, batch):
        xq = batch['xq_padded'] # n_class * n_query * max_len * mfcc_dim
        xs = batch['xs_padded'] # n_class * n_support * max_len * mfcc_dim
        xq_len = batch['xq_len'] # n_class * n_query 
        xs_len = batch['xs_len'] # n_class * n_support
        
        assert xq.shape[0] == xq_len.shape[0]
        assert xs.shape[0] == xs_len.shape[0]
        
        n_class = xq_len.shape[0]
        n_query = xq_len.shape[1]
        n_support = xs_len.shape[1]

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
            
        seq_len = torch.cat([xq_len.view(n_class * n_query, -1).squeeze(-1),
                            xs_len.view(n_class * n_support, -1).squeeze(-1)], 0)
        seq_len = Variable(seq_len, requires_grad=False)

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), 
                        xq.view(n_class * n_query, *xq.size()[2:])],
                        0)
        
        _len, perm_idx = seq_len.sort(0, descending=True)

        x = x[perm_idx]

        packed_input = pack_padded_sequence(x, _len.cpu().numpy().astype(dtype=np.int32), batch_first=True)

        packed_output, _ = self.encoder_rnn.forward(packed_input)

        z, _ = pad_packed_sequence(packed_output, batch_first=True)

        _, unperm_idx = perm_idx.sort(0)
        z = z[unperm_idx]

        #z, _ = self.encoder_rnn.forward(x)

        z = self.encoder_linear.forward((z, seq_len))

        z_dim = z.size(-1)
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]


        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze(-1)).float().mean()

        logger.info(f'loss: {loss_val.item()}, acc: {acc_val.item()}')

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def evaluate(self, batch):
        return self.loss(batch)
        

@register_model('few_shot_speech_var_len')
def load_few_short_speech(**kwargs):
    in_dim = kwargs['in_dim'][-1]
    out_dim = kwargs['out_dim']
    n_rnn = kwargs['n_rnn']
    '''
    encoder_rnn = nn.Sequential(
                    nn.LSTM(in_dim, out_dim,
                    bidirectional=True,
                    batch_first=True)
                )
    '''
    encoder_rnn = nn.LSTM(in_dim, out_dim, n_rnn,
                    bidirectional=True,
                    batch_first=True)
    encoder_linear = nn.Sequential(
            TemoralAvgPooling(),
            nn.Linear(out_dim * 2, out_dim, bias=True),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim, bias=True),
            nn.Tanh(),
            nn.Sigmoid()
            # UnitNormalize()
        )
    
    gpu_num = kwargs['gpu_num']
    if gpu_num > 1:
        logger.info(f'Using multiple GPUs: {gpu_num}')
        device_ids = list(range(gpu_num))

        encoder_rnn = nn.DataParallel(
            encoder_rnn,
            device_ids=device_ids
        )
        encoder_linear = nn.DataParallel(
            encoder_linear,
            device_ids=device_ids
        )
        return FewShotSpeech(encoder_rnn, encoder_linear)

    else:
        logger.info('Using single GPU')
        return FewShotSpeech(encoder_rnn, encoder_linear)

