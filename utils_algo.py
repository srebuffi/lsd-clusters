import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class BCE(nn.Module):
    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


class BCE_softlabels(nn.Module):
    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)

        P = prob1.mul_(prob2)
        P = P.sum(1)
        neglogP = - (simi * torch.log(P + BCE.eps) + (1. - simi) * torch.log(1. - P + BCE.eps))
        return neglogP.mean()


def PairEnum(x):
    """ Enumerate all pairs of feature in x."""
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2
