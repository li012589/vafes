import torch
import numpy as np
from .transformation import CouplingBijector


class SplineFlow(CouplingBijector):
    r'''
    Basic class for spline-based normalizing flow, e.g., neural spline flow, cubic spline flow.
    '''
    def __init__(self, name="SplineFlow"):
        super().__init__(name)

    @staticmethod
    def _parameterIter(inverse, maskList, kwargs, n):
        r'''
        manage safe boundaries
        Args:
            maskList(ndarray): mask is used to compute the number of transformations, so a minimum safe distance is achieve;
        '''
        # use of boundaryList
        if kwargs.get('boundaryList') is not None:
            kwargs['boundary'] = kwargs['boundaryList'][n]
        return kwargs

    @staticmethod
    def _coupling(inverse, x, params, sections, spline, boundary, indentation, eps, minLog, splineParams, linearBound, *args, **kwargs):
        r'''
        the coupling of spline flow.
        Args:
            inverse, x, params: arguemnts from template;
            sections (list of int): the shape list to split the output from the network;
            spline (utils.SplineFn): the spline used;
            boundary (tuple of ndarray): the four boundaries (left, right, bottom, top);
            indentation (float): the safe boundary dead zone for each transformation;
            eps (float): the minimum value of gradient for numerical stability;
            minLog (float): the minimum log det jacobian for numerical stability;
            splineParams: arguments for the spline, except these from the neuralnetwors.
        '''
        params = params[0].reshape(x.shape[0], np.sum(sections), -1).transpose(1, -1)[:, :x.shape[-1], :]
        params = torch.split(params, sections, dim=-1)

        if linearBound:
            ld = torch.zeros_like(x)
            if inverse:
                inBoundMask = (x >= boundary[-2]) & (x <= boundary[-1])
                _x = x[inBoundMask]
                x[inBoundMask], ld[inBoundMask] = spline.inversenGrad(_x, tuple(term[inBoundMask] for term in params) + splineParams, boundary, *args, **kwargs)
            else:
                inBoundMask = (x >= boundary[0]) & (x <= boundary[1])
                _x = x[inBoundMask]
                x[inBoundMask], ld[inBoundMask] = spline.forwardnGrad(_x, tuple(term[inBoundMask] for term in params) + splineParams, boundary, *args, **kwargs)
        else:
            params = params + splineParams
            if inverse:
                x, ld = spline.inversenGrad(x, params, boundary, *args, **kwargs)
            else:
                x, ld = spline.forwardnGrad(x, params, boundary, *args, **kwargs)

        ld = torch.clip(ld, min=0.0) + eps
        ld = torch.clip(torch.log(ld), min=minLog)
        return x, ld

    @classmethod
    def initalize(cls, sections=None, spline=None, boundary=(-1, 1, -1, 1), boundaryList=None, indentation=None, minLog=-50, eps=1e-7, splineAllParams=None, linearBound=False, **kwargs):
        r'''
        initalize parameters
        '''
        if splineAllParams is None:
            splineAllParams = spline.initalize()
        return super().initalize(**kwargs) | {'sections': sections, 'spline': spline, 'boundary': boundary, 'boundaryList': boundaryList, 'indentation': indentation, 'eps': eps, 'minLog': minLog,  'linearBound': linearBound} | splineAllParams
