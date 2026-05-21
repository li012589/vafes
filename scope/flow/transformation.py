import numpy as np
import torch
from torch import nn


class Bijector(nn.Module):
    r'''
    Template for bijective transformations.
    '''
    def __init__(self, name="Bijector"):
        super().__init__()
        self.name = name

    @classmethod
    def forward(cls, z, T, mask=None, inShape=None, outShape=None, *args, **kargs):
        r'''
        the forward transformation of the bijector.
        Args:
            z (ndarray): the variable to be transformated;
            T (float or ndarry): the temperature factor;
            mask (ndarray): the mask to select part of z to perform transformation, default None, means use the whole z;
            inShape (list of int): the shape list used to reshape the input, default None, means no reshape is performed;
            outShape (list of int): the shape list used to reshape the output, default None, means no reshape is performed;
            args, kwargs: arguments for cls.bijection.
        '''
        if mask is not None:
            _z = torch.masked_select(z, mask.bool())
            _z, logDet = cls.bijection(False, _z, T, *args, **kargs)
            z = z.masked_scatter(mask.bool(), _z)
        else:
            z, logDet = cls.bijection(False, z, T, *args, **kargs)
        if outShape is not None:
            return z.reshape(z.shape[0], *outShape), logDet
        else:
            return z, logDet

    @classmethod
    def inverse(cls, x, T, mask=None, inShape=None, outShape=None, *args, **kargs):
        r'''
        the forward transformation of the bijector.
        Args:
            x (ndarray): the variable to be transformated;
            T (float or ndarry): the temperature factor;
            mask (ndarray): the mask to select part of z to perform transformation, default None, means use the whole z;
            reshape (list of int): the shape list used to reshape the output, default None, means no reshape is performed;
            args, kwargs: arguments for cls.bijection.
        '''
        if inShape is not None:
            x = x.reshape(x.shape[0], *inShape)
        if mask is not None:
            _x = torch.masked_select(x, mask.bool())
            _x, logDet = cls.bijection(True, _x, T, *args, **kargs)
            x = x.masked_scatter(mask.bool(), _x)
        else:
            x, logDet = cls.bijection(True, x, T, *args, **kargs)
        return x, logDet

    @staticmethod
    def bijection(inverse, x, T, **parameters):
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def initalize(mask=None, inShape=None, outShape=None):
        r'''
        initalize parameters
        '''
        return {'mask': mask, 'inShape': inShape, 'outShape': outShape}


class CouplingBijector(Bijector):
    r'''
    Template for bijective neural network transformation via coupling transformation.
    '''
    def __init__(self, name="NeuralBijector"):
        super().__init__(name)

    @classmethod
    def bijection(cls, inverse, z, T, maskList, maskConpList, networkList, **kwargs):
        r'''
        the coupling transformation.
        Args:
            inverse (bool): the forward or the inverse transformation;
            z (ndarray): the variables to be transformated;
            T (float or ndarray): the temperature factor;
            maskList (list of mask): used to filter the part of variables that are changed;
            maskConpList (list mask): used to filer the part of variable that are kept unchanged, usually conpensatory of maskList;
            networkList (list of torch.Module): apply on the unchanged variables to provide the transformation parameters;
            kwargs: parameters for cls._coupling
        '''
        if isinstance(T, torch.Tensor):
            if len(T.shape) == 0:
                T = T.reshape(1, 1)
            else:
                T = T.reshape(T.shape[0], -1)
        else:
            T = torch.Tensor([[T]]).to(z)

        logDet = 0
        if inverse:
            idx = reversed(range(len(networkList)))
        else:
            idx = range(len(networkList))
        for i in idx:
            maskConp = maskConpList[i].bool()
            mask = maskList[i].bool()
            lower = torch.masked_select(z, mask).reshape(z.shape[0], -1)
            upper = torch.masked_select(z, maskConp).reshape(*z.shape[:-1], int(maskConp.sum().item()//np.prod(maskConp.shape[:-1])))
            if T.shape[0] > 1:
                _T = T.view(-1, *([1] * (len(upper.shape) - 2)), T.shape[-1]).repeat(1, 1, *(upper.shape[2:]))
            else:
                _T = T.repeat(upper.shape[0], 1, *upper.shape[2:])
            upper = torch.cat([upper, _T], dim=1)

            kwargs = cls._parameterIter(inverse, maskList, kwargs, i)
            param = [network(upper) for network in networkList[i]]
            lower, _logDet = cls._coupling(inverse, lower, param, **kwargs)

            logDet += _logDet.reshape(upper.shape[0], -1).sum(1, keepdim=True)
            z = z.masked_scatter(mask, lower)
        return z, logDet

    @staticmethod
    def _parameterIter(inverse, maskList, kwargs, n):
        r'''
        modfiy the arguments for _coupling at each iteration.
        '''
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def _coupling(inverse, x, param, **kwargs):
        r'''
        the actual coupling transformation.
        '''
        raise NotImplementedError(str(type(self)))

    @classmethod
    def initalize(cls, maskList=None, maskConpList=None, networkList=None, **kwargs):
        r'''
        initalize parameters
        '''
        if maskConpList is None:
            maskConpList = 1 - maskList
        return super().initalize(**kwargs) | {'maskList': maskList, 'maskConpList': maskConpList, 'networkList': networkList}
