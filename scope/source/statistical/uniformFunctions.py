import torch
import numpy as np


def uniformSample(batch, nvars, T, low, high, dtype=None, device=None):
    r'''
    Sampling of uniform distribution.
    Args:
        batch (int): leading batch dimensions of the samples;
        nvars (list of int): dimensions of the distribution;
        T (float or ndarray of shape [batch]): the temperature factor;
        low (float, or ndarray of shape nvars): lowest value allowed;
        high (float, or ndarray of shape nvars): highest value allowed;
        dtype: the date type, default follows low or high, if both is not ndarray, use torch.float32.
        device: the device, default follows low or high, if both is not ndarray, use cpu.
    '''
    if nvars is None:
        nvars = list(low.shape)
    if dtype is None:
        if isinstance(low, torch.Tensor):
            dtype = low.dtype
        elif isinstance(high, torch.Tensor):
            dtype = high.dtype
        else:
            dtype = torch.float32
    if device is None:
        if isinstance(low, torch.Tensor):
            device = low.device
        elif isinstance(high, torch.Tensor):
            device = high.device
        else:
            device = torch.device('cpu')

    return torch.rand([batch] + nvars, dtype=dtype).to(device=device) * (high - low) + low


def uniformEnergy(x, low, high, outBoundE=1e12):
    r'''
    Compute the energy.
    Args:
        x (ndarray): samples;
        low (float, or ndarray of shape nvars): lowest value allowed;
        high (float, or ndarray of shape nvars): highest value allowed;
        outBoundE: the energy assigned to sample outside boundary, default inf.
    '''
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high)
    lb = low.le(x)
    ub = high.gt(x)
    ob = lb.mul(ub)
    ob = (~ob).reshape(x.shape[0], -1).sum(-1, keepdim=True) * outBoundE
    return ob


def uniformLogPartition(nvars, T, low, high):
    r'''
    compute the log partition function.
    Args:
        batch (int): leading batch dimensions of the samples;
        nvars (list of int): dimensions of the distribution;
        T (float or ndarray of shape [batch]): the temperature factor;
        low (float, or ndarray of shape nvars): lowest value allowed;
        high (float, or ndarray of shape nvars): highest value allowed;
    '''
    if nvars is None:
        nvars = [1]
    if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
        logZ = torch.log(high - low)
        if np.prod(logZ.shape) > 1:
            logZ = logZ.reshape(-1, np.prod(nvars)).sum(-1, keepdim=True)
        else:
            logZ = logZ * np.prod(nvars)
    else:
        logZ = np.log(high - low) * np.prod(nvars)
    return logZ
