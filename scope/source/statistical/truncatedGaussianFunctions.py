import torch
import numpy as np
from .gaussianFunctions import gaussianEnergy


def truncatedGaussianSample(batch, nvars, T, mu, logsigma, low, high):
    r'''
    Sampling process of uncorrelated truncated gaussian using bijections. This approach is fast and autodiff-friendly, but maybe numerically unstable due to erf at large values.
    Args:
        batch (int): leading batch dimensions of the samples;
        nvars (list of int): dimensions of the distribution;
        T (float or ndarray of shape [batch]): the temperature factor;
        mu (ndarray of shape nvars): parameters of the distribution, the mean pos of samples;
        logsigma (ndarray of shape nvars): parameters of the distribution, controls the variance of samples;
        low (float, or ndarray of shape nvars): lowest value allowed;
        high (float, or ndarray of shape nvars): highest value allowed;
    '''
    if nvars is None:
        nvars = list(mu.shape)
    if isinstance(T, torch.Tensor):
        sqrtT = torch.sqrt(2 * T.view(-1, *[1] * len(nvars)))
    else:
        sqrtT = np.sqrt(2 * T)

    size = [batch] + nvars

    zhigh = torch.clip(
        torch.erf((high - mu) * torch.exp(-logsigma) / sqrtT),
        max=1) - 1e-7
    zlow = torch.clip(
        torch.erf((low - mu) * torch.exp(-logsigma) / sqrtT),
        min=-1) + 1e-7

    # when zhigh is too close to zlow, ie sigma is big, use rand instead.
    uniformMask = (zhigh - zlow) < 1e-5

    _z = torch.rand(size).to(mu)
    z = _z * zhigh + (1 - _z) * zlow

    invZ = torch.erfinv(z)
    samples = torch.clip(invZ * torch.exp(logsigma) * sqrtT + mu, low, high)
    uniformSamples = torch.masked_select(_z, uniformMask) * torch.masked_select(high, uniformMask) + (1 - torch.masked_select(_z, uniformMask)) * torch.masked_select(low, uniformMask)
    samples = samples.masked_scatter(uniformMask, uniformSamples)
    return samples


def truncatedGaussianEnergy(x, mu, logsigma, low, high):
    r'''
    compute the energy of x.
    Args:
        x (ndarray): samples;
        mu (ndarray of shape nvars): parameters of the distribution, the mean pos of samples;
        logsigma (ndarray of shape nvars): parameters of the distribution, controls the variance of samples;
        low (float, or ndarray of shape nvars): lowest value allowed;
        high (float, or ndarray of shape nvars): highest value allowed;
    '''
    assert torch.all(x <= high) and torch.all(x >= low)
    return gaussianEnergy(x, mu, logsigma)


def truncatedGaussianLogPartition(nvars, T, mu, logsigma, low, high):
    r'''
    compute the log partition function.
    Args:
        nvars (list of int): dimensions of the distribution;
        T (float or ndarray of shape [batch]): the temperature factor;
        mu (ndarray of shape nvars or [batch, *nvars]): parameters of the distribution, the mean pos of samples;
        logsigma (ndarray of shape nvars or [batch, *nvars]): parameters of the distribution, controls the variance of samples;
        low (float, or ndarray of shape nvars or [batch, *nvars]): lowest value allowed;
        high (float, or ndarray of shape nvars or [batch, *nvars]): highest value allowed;
    '''
    if nvars is None:
        nvars = list(mu.shape)
    if isinstance(T, torch.Tensor):
        T = T.view([-1] + len(nvars) * [1])
        sqrtT = torch.sqrt(T)
        sqrt2T = torch.sqrt(2 * T)
        logT = torch.log(T)
    else:
        sqrtT = np.sqrt(T)
        sqrt2T = np.sqrt(2 * T)
        logT = np.log(T)
    CDFdiff = 0.5 * (torch.erf((high - mu) * torch.exp(-logsigma) / sqrt2T) - torch.erf((low - mu) * torch.exp(-logsigma) / sqrt2T))

    return torch.log(CDFdiff).reshape(-1, np.prod(nvars)).sum(-1, keepdim=True) + 0.5 * np.prod(nvars) * np.log(2. * np.pi) + (logsigma + 0.5 * logT).reshape(-1, np.prod(nvars)).sum(-1, keepdim=True)
