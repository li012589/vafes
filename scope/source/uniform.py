import torch
from torch import nn
from .distribution import Distribution
from .statistical import uniformSample, uniformEnergy, uniformLogPartition


class Uniform(Distribution):
    def __init__(self, name="uniform"):
        super().__init__(name)

    @staticmethod
    def sample(batch, nvars=None, T=1.0, low=None, high=None, outBoundE=1e12):
        return uniformSample(batch, nvars, T, low, high)

    @staticmethod
    def energy(x, low=None, high=None, outBoundE=1e12):
        return uniformEnergy(x, low, high, outBoundE)

    @staticmethod
    def logPartition(nvars=None, T=1.0, low=None, high=None, outBoundE=1e12):
        return uniformLogPartition(nvars, T, low, high)

    @staticmethod
    def initalize(dic):
        r'''
        initalize parameters for unifrom distribution.
        Args:
           low (float or ndarray): the lowest value;
           high (float or ndarray): the highest value;
        '''
        low = dic.get('low', None)
        high = dic.get('high', None)

        if low is None:
            low = -1
        if high is None:
            high = 1
        if not isinstance(low, nn.Parameter):
            if isinstance(low, torch.Tensor):
                dic['low'] = nn.Parameter(low.detach(), requires_grad=False)
            else:
                dic['low'] = nn.Parameter(torch.tensor(low), requires_grad=False)
        if not isinstance(high, nn.Parameter):
            if isinstance(high, torch.Tensor):
                dic['high'] = nn.Parameter(high.detach(), requires_grad=False)
            else:
                dic['high'] = nn.Parameter(torch.tensor(high), requires_grad=False)
        return dic

