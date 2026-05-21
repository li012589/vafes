import torch
from torch import nn
from .distribution import Distribution
from .statistical import truncatedGaussianSample, truncatedGaussianEnergy, truncatedGaussianLogPartition


class TruncatedGaussian(Distribution):
    def __init__(self, name="truncatedGaussian"):
        super().__init__(name)

    @staticmethod
    def sample(batch, nvars=None, T=1.0, mu=None, logsigma=None, low=None, high=None, eps=1e-5):
        return truncatedGaussianSample(batch, nvars, T, mu, logsigma, low+eps, high-eps)

    @staticmethod
    def energy(x, mu, logsigma, low, high):
        return truncatedGaussianEnergy(x, mu, logsigma, low, high)

    @staticmethod
    def logPartition(nvars=None, T=1.0, mu=None, logsigma=None, low=None, high=None):
        return truncatedGaussianLogPartition(nvars, T, mu, logsigma, low, high)

    @staticmethod
    def initalize(dic, trainable=True):
        r'''
        initalize parameters for truncated guassian distributions
        Args:
            low (float or ndarray): the lowest value;
            high (float or ndarray): the highest value;
            nvars (list of int): the dimension of samples.
        '''
        nvars = dic.pop('nvars', None)
        low = dic.get('low', None)
        high = dic.get('high', None)
        mu = dic.get('mu', None)
        logsigma = dic.get('logsigma', None)

        if mu is None:
            mu = torch.randn(nvars) * 0.1
        if not isinstance(mu, nn.Parameter):
            dic['mu'] = nn.Parameter(mu, requires_grad=trainable)
        if logsigma is None:
            logsigma = torch.randn(nvars) * 0.05
        if not isinstance(logsigma, nn.Parameter):
            dic['logsigma'] = nn.Parameter(logsigma, requires_grad=trainable)
        if low is None:
            low = -1
        if high is None:
            high = 1
        if not isinstance(low, nn.Parameter):
            dic['low'] = nn.Parameter(torch.tensor(low), requires_grad=False)
        if not isinstance(high, nn.Parameter):
            dic['high'] = nn.Parameter(torch.tensor(high), requires_grad=False)
        return dic
