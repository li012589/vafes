import torch
from torch import nn


class SplineFn(nn.Module):
    '''
    Functional template for spline interpolation.
    '''
    def __init__(self, name="FunctionalSpline"):
        super().__init__()

    @classmethod
    def forward(cls, x, params, boundary, *args, **kwargs):
        '''
        compute the outputs of the spline.
        Args:
            x (ndarray, batch x 1): input tensor;
            params (list): list of initial parameters;
            boundary (tuple of ndarray): the four boundaries (left, right, bottom, top);
            args, kwargs: additional options.
        '''
        params, binCumWidth, binWidth, _, _ = cls._preprocess(params)
        left, right, bottom, top = boundary
        x = (x - left) / (right - left)
        binId = torch.searchsorted(binCumWidth, x.unsqueeze(-1), right=True) - 1
        binId = torch.clip(binId, max=binCumWidth.shape[-1] - 2)
        leftBase = binCumWidth.gather(-1, binId)[..., 0] # subtract cumulative base so x starts from 0
        width = binWidth.gather(-1, binId)[..., 0] # shrink the value so lamd in [0, 1]
        lamd = (x - leftBase) / width
        y = cls._forward(lamd, [term.gather(-1, binId)[..., 0] for term in params], *args, **kwargs)
        return y * (top - bottom) + bottom

    @classmethod
    def forwardnGrad(cls, x, params, boundary, *args, **kwargs):
        '''
        compute the outputs and gradient of the spline.
        Args:
            x (ndarray, batch x 1): input tensor;
            params (list): list of initial parameters;
            boundary (tuple of ndarray): the four boundaries (left, right, bottom, top);
            args, kwargs: additional options.
        '''
        params, binCumWidth, binWidth, _, _ = cls._preprocess(params)
        left, right, bottom, top = boundary
        x = (x - left) / (right - left)
        binId = torch.searchsorted(binCumWidth, x.unsqueeze(-1), right=True) - 1
        binId = torch.clip(binId, max=binCumWidth.shape[-1] - 2)
        leftBase = binCumWidth.gather(-1, binId)[..., 0] # subtract cumulative base so x starts from 0
        width = binWidth.gather(-1, binId)[..., 0] # shrink the value so lamd in [0, 1]
        lamd = (x - leftBase) / width
        if hasattr(cls, '_forwardnGrad'):
            y, grad = cls._forwardnGrad(lamd, [term.gather(-1, binId)[..., 0] for term in params], *args, **kwargs)
        else:
            y, grad = cls._forward(lamd, [term.gather(-1, binId)[..., 0] for term in params], *args, **kwargs), cls._grad(lamd, [term.gather(-1, binId)[..., 0] for term in params], *args, **kwargs)
        return y * (top - bottom) + bottom, grad * (top - bottom) / (right - left) / width

    @classmethod
    def grad(cls, x, params, boundary, *args, **kwargs):
        '''
        compute the gradients of the spline, w.r.t. inputs.
        Args:
            x (ndarray, batch x 1): input tensor;
            params (list): list of initial parameters;
            boundary (tuple of ndarray): the four boundaries (left, right, bottom, top);
            args, kwargs: additional options.
        '''
        params, binCumWidth, binWidth, _, _ = cls._preprocess(params)
        left, right, bottom, top = boundary
        x = (x - left) / (right - left)
        binId = torch.searchsorted(binCumWidth, x.unsqueeze(-1), right=True) - 1
        binId = torch.clip(binId, max=binCumWidth.shape[-1] - 2)
        leftBase = binCumWidth.gather(-1, binId)[..., 0] # subtract cumulative base so x starts from 0
        width = binWidth.gather(-1, binId)[..., 0] # shrink the value so lamd in [0, 1]
        lamd = (x - leftBase) / width
        grad = cls._grad(lamd, [term.gather(-1, binId)[..., 0] for term in params], *args, **kwargs)
        return grad * (top - bottom) / (right - left) / width

    @classmethod
    def inverse(cls, y, params, boundary, *args, **kwargs):
        '''
        compute the inverse of the spline, given the output values (y).
        Args:
            y (ndarray, batch x 1): input tensor;
            params (list): list of initial parameters;
            boundary (tuple of ndarray): the four boundaries (left, right, bottom, top);
            args, kwargs: additional options.
        '''
        params, binCumWidth, binWidth, binCumHeight, _ = cls._preprocess(params)
        left, right, bottom, top = boundary
        y = (y - bottom) / (top - bottom)
        binId = torch.searchsorted(binCumHeight, y.unsqueeze(-1), right=True) - 1
        binId = torch.clip(binId, min=0, max=binCumHeight.shape[-1] - 2)
        leftBase = binCumWidth.gather(-1, binId)[..., 0]
        width = binWidth.gather(-1, binId)[..., 0]
        lamd = cls._inverse(y, [term.gather(-1, binId)[..., 0] for term in params], *args, **kwargs)
        x = lamd * width + leftBase
        return x * (right - left) + left

    @classmethod
    def inversenGrad(cls, y, params, boundary, *args, **kwargs):
        '''
        compute the inverse of the spline, given the output values (y), and inverse gradient (w.r.t. y) of the spline (1 / (grad w.r.t. x).)
        Args:
            y (ndarray, batch x 1): input tensor;
            params (list): list of initial parameters;
            boundary (tuple of ndarray): the four boundaries (left, right, bottom, top);
            args, kwargs: additional options.
        '''
        params, binCumWidth, binWidth, binCumHeight, _ = cls._preprocess(params)
        left, right, bottom, top = boundary
        y = (y - bottom) / (top - bottom)
        binId = torch.searchsorted(binCumHeight, y.unsqueeze(-1), right=True) - 1
        binId = torch.clip(binId, min=0, max=binCumHeight.shape[-1] - 2)
        leftBase = binCumWidth.gather(-1, binId)[..., 0]
        width = binWidth.gather(-1, binId)[..., 0]

        params = [term.gather(-1, binId)[..., 0] for term in params]
        if hasattr(cls, '_inversenGrad'):
            lamd, grad = cls._inversenGrad(y, params, *args, **kwargs)
        else:
            lamd = cls._inverse(y, params, *args, **kwargs)
            grad = 1 / (cls._grad(lamd, params, *args, **kwargs) + 1e-10)

        x = lamd * width + leftBase
        return x * (right - left) + left, grad / (top - bottom) * (right - left) * width

    @staticmethod
    def _preprocess(params, inverse=False):
        '''
        Function template, convert input params into params of each segments.
        Args:
            params (list): list of initial parameters;

        Returns:
            params (list of ndarray with shape of [batch, nbins]): parameters for each segments.
            binCumWidth (ndarray): accumulated bin width location in the x-axis;
            binWidth (ndarray):  bin width in the x-axis;
            binCumHeight (ndarray): accumulated bin height location in the y-axis.
            binCumHeight (ndarray): bin height in the y-axis.
        '''
        binCumWidth = params[0]
        binWidth = params[1]
        binCumHeight = params[2]
        binHeight = params[3]
        return params, binCumWidth, binWidth, binCumHeight, binHeight

    @staticmethod
    def _forward(lamd, params, *args, **kwargs):
        raise NotImplementedError("_forward not implemented")

    @staticmethod
    def _grad(x, params, *args, **kwargs):
        raise NotImplementedError("_grad not implemented")

    @classmethod
    def _inverse(cls, y, params, *args, **kwargs):
        raise NotImplementedError("_inverse not implemented")

    @staticmethod
    def initalize(*args, **kwargs):
        r'''
        initalize parameters
        '''
        raise NotImplementedError("SplineFn.initalize is not implemented")
