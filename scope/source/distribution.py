import torch
from torch import nn


class Distribution(nn.Module):
    r'''
    The template for probability distribution.
    '''
    def __init__(self, name="Distribution"):
        super().__init__()
        self.name = name

    @classmethod
    def sample(cls, batch, nvars=None, T=1.0, **parameters):
        r'''
        Sample method template
        Args:
            batch (int): leading batch dimensions of the samples;
            nvars (list of int): dimensions of the distribution;
            T (float or ndarray of shape [batch], default 1.0): the temperature factor;
            params (dict): default None, other parameters.
        '''
        raise NotImplementedError(f"{cls.__name__}.sample is not implemented")

    @classmethod
    def logProbability(cls, x, T=1.0, **parameters):
        r'''
        compute the log probability of the sample x. Please be awared that the probability MAY NOT be normalized.
        Args:
            x (ndarray): samples with batch dim;
            T (float or ndarray of shape [batch], default 1.0): the temperature factor;
            params (list): default None, other parameters.
        '''
        nvars = x.shape[1:]
        if hasattr(cls, "energyWithT"):
            energy = cls.energyWithT(x, T, **parameters)
        else:
            if isinstance(T, torch.Tensor):
                T = T.reshape(-1, 1)
            energy = cls.energy(x, **parameters) / T
        return -(energy + cls.logPartition(nvars, T, **parameters))

    @staticmethod
    def energy(x, **parameters):
        r'''
        compute the (usually unnormalized) energy of the sample x.
        Args:
            x (ndarray): samples with batch dim;
            params (list): default None, other parameters.
        '''
        raise NotImplementedError("Distribution.energy is not implemented")

    @staticmethod
    def logPartition(nvars=None, T=1.0, **parameters):
        r'''
        compute the log partition of the sample x, this method maynot exist due to integrable problem. By default, assuming the probability doesn't depend on T with the partition function being 1.
        Args:
            T (float or ndarray of shape [batch], default 1.0): the temperature factor;
            params (list): default None, other parameters.
        '''
        return 0

    @staticmethod
    def initalize(*args, **kwargs):
        r'''
        initalize the parameters. Return dict contains all arguments.
        '''
        raise NotImplementedError("Distribution.initalize is not implemented")


class TransformedDistribution(Distribution):
    r'''
    The template for probability distribution ansatz, with parameterized transformations.
    '''
    def __init__(self, name="TransformedDistribution"):
        super().__init__(name)

    @staticmethod
    def forward(z, T=1.0, transformationList=None, transformationParamList=None):
        r'''
        The forward transformation of the bijection, and its log jacobian.
        Args:
            z (ndarray): input sample with batch dim;
            T (float or ndarray of shape [batch], default 1.0): the temperature factor;
            transformationList (list of Bijector): the transformations to perform;
            transformationParamList (list of dict): the arguments for each transformations;
        '''
        logDet = 0
        for n in range(len(transformationList)):
            z, _logDet = transformationList[n].forward(z, T, **transformationParamList[n])
            logDet += _logDet.view(-1, 1)
        return z, logDet

    @staticmethod
    def inverse(x, T=1.0, transformationList=None, transformationParamList=None):
        r'''
        The forward transformation of the bijection, and its log jacobian.
        Args:
            x (ndarray): input sample with batch dim;
            T (float or ndarray of shape [batch], default 1.0): the temperature factor;
            transformationList (list of Bijector): the transformations to perform;
            transformationParamList (list of dict): the arguments for each transformations;
        '''
        logDet = 0
        for n in reversed(range(len(transformationList))):
            x, _logDet = transformationList[n].inverse(x, T, **transformationParamList[n])
            logDet += _logDet.view(-1, 1)
        return x, logDet

    @classmethod
    def samplenProb(cls, batch, nvars=None, T=1.0, transformationList=None, transformationParamList=None, prior=None, priorParams=None):
        r'''
        Sample method for normalizing flow model. Also provide the log probability.
        Args:
            batch (int or list of int): leading batch dimensions of the samples;
            nvars (list of int): dimensions of the distribution, default None meaning nvars should be implied by parameters;
            T (float or ndarray of shape [batch], default 1.0): the temperature factor;
            transformationList (list of Bijector): the transformations to perform;
            transformationParamList (list of dict): the arguments for each transformations;
            prior (Distribution): the prior distribution;
            priorParams (dict): the arguements for prior.
        '''
        if nvars is None:
            z = prior.sample(batch, transformationParamList[0]['inShape'], T, **priorParams)
        else:
            z = prior.sample(batch, nvars, T, **priorParams)
        logp = prior.logProbability(z, T, **priorParams)
        x, logjac = cls.forward(z, T, transformationList, transformationParamList)
        return x, logp - logjac

    @classmethod
    def sample(cls, *args, **kwargs):
        r'''
        Sample method for normalizing flow model.
        '''
        return cls.samplenProb(*args, **kwargs)[0]

    @classmethod
    def energyWithT(cls, x, T=1.0, transformationList=None, transformationParamList=None, prior=None, priorParams=None):
        r'''
        compute energy (logProbability) of the sample x.
        Args:
            x (ndarray): input sample;
            T (float, ndarray, None, default None): temperature factor, None means not use temperature arguement in forward/inverse, and T = 1.0 for prior.
        '''
        z, logjac = cls.inverse(x, T, transformationList, transformationParamList)
        if hasattr(prior, "energyWithT"):
            energy = prior.energyWithT(z, T, **priorParams)
        else:
            if isinstance(T, torch.Tensor):
                T = T.reshape(-1, 1)
            energy = prior.energy(z, **priorParams) / T
        return energy - logjac
