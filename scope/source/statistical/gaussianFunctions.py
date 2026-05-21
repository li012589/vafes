import torch
import numpy as np


def gaussianSample(batch, nvars, T, mu, logsigma):
    r'''
    Sampling process of uncorrelated gaussian.
    Args:
        batch (int): leading batch dimensions of the samples;
        nvars (list of int): dimensions of the distribution;
        T (float or ndarray of shape [batch]): the temperature factor;
        mu (ndarray of shape nvars): parameters of the distribution, the mean pos of samples;
        logsigma (ndarray of shape nvars): parameters of the distribution, controls the variance of samples;
    '''
    if nvars is None:
        nvars = list(mu.shape)
    size = [batch] + nvars
    if isinstance(T, torch.Tensor):
        sqrtT = torch.sqrt(T.view(-1, *[1] * len(nvars)))
    else:
        sqrtT = np.sqrt(T)
    return (torch.randn(size, dtype=logsigma.dtype).to(logsigma) * torch.exp(logsigma) * sqrtT + mu)


def gaussianEnergy(x, mu, logsigma):
    r'''
    compute the energy of x.
    Args:
        x (ndarray): samples;
        mu (ndarray of shape nvars): parameters of the distribution, the mean pos of samples;
        logsigma (ndarray of shape nvars): parameters of the distribution, controls the variance of samples;
    '''
    return 0.5 * ((x - mu)**2 * torch.exp(-2 * logsigma)).reshape(x.shape[0], -1).sum(dim = 1, keepdim=True)
