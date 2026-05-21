from functools import wraps

import torch


def makePairPotential(idx=None, cutoffRange=0, eps=1e-5, max=1e8):
    r'''
    A decorator factory for creating potential from pair interations.

    There are three decorations: 1. indexing using idx; 2. cut off too-closed pairs using cutoffRange; 3. clip all distances to a minimum of eps (in case cut-off to zero is not preferred). They're also applied in that order. For the return value from the pair interation function, a clip is applied to keep below a maximum potential value.

    Args:
        idx (ndarray, [Npair, 2] or None): indices of the 2 atoms in each pair, None means all pairs are invloved;
        cutoffRange (int or ndarray, [batch, Npair]): the cutoff range to ignore too-close interations;
        eps (float, default 1e-7): the eps to avoid zero norm;
        max (float, default 1e8): maximum value of coulomb.
    Return:
        wrapper function for pair interation functions
    '''
    def decorator(func):
        r'''
        A decorator for creating potential from pair interactions.
        Args:
            func (tuple of ndarray, Tuple([batch, Npair])) -> (ndarray, [batch, 1]): the function to compute pair interaction energy.
        Return:
            (tuple of ndarray, [batch, N, N] + Tuple([batch, Npair])) -> (ndarray, [batch, 1]): potential function that takes the distance matrix and interaction parameters for each pair to compute the potential values.
        '''
        @wraps(func)
        def wrapper(*args):
            r'''
            Potential function created from a pair interaction.
            Args:
                args (tuple of ndarray, [batch, N, N] + Tuple([batch, Npair])): the input arguments. The first one must be the distance matrix, the results are parameters for each interaction pairs.
            Return:
                (ndarray, [batch, 1]): the potential value.
            '''
            # select pairs from distance
            if idx is not None:
                distance = args[0][:, idx[:, 0], idx[:, 1]]
            else:
                distance = args[0].reshape(args[0].shape[0], -1)
            allowed = distance > cutoffRange

            if isinstance(cutoffRange, torch.Tensor):
                rIJ = torch.clip(distance * allowed, min=eps).reshape(distance.shape[0], -1)
                params = [(term * allowed).reshape(allowed.shape[0], -1) for term in args[1:]]
            else:
                rIJ = torch.clip(torch.masked_select(distance, allowed), min=eps).reshape(distance.shape[0], -1)
                params = [torch.masked_select(term, allowed[:term.shape[0]]).reshape(term.shape[0], -1) for term in args[1:]]
            return torch.clip(func(rIJ, *params),  max=max).sum(-1, keepdim=True)
        return wrapper
    return decorator


def makeVecBasedPotential(idx, termComputeFunc, eps=1e-5, max=1e8):
    r'''
    A decorator factory for creating potential from interations of multiple vectors bewteen atoms (e.g., angles and dihedrals).

    The procedure is as follows, 1. indexing vectors from the relative vector matrix; 2. compute the term used in the interation (e.g., angles or dihedrals); 3. clip the interaction to a maximum of max.

    Args:
        idx(ndarray, [Nterm, M]): indices of M atoms that forms the vectors used in computing the term of the interation (e.g., M=3 for angles and M=4 for dihedrals);
        eps (float, default 1e-7): the eps to avoid zero norm;
        max (float, default 1e8): maximum value of coulomb.
    Return:
        wrapper function for pair interation functions
    '''
    def decorator(func):
        r'''
        A decorator for creating potential from interactions.
        Args:
            func (tuple of ndarray, Tuple([batch, Nterm])) -> (ndarray, [batch, 1]): the function to compute interaction energy using terms (e.g., angles or dihedrals).
        Return:
            (tuple of ndarray, [batch, N, 3] + Tuple([batch, Nterm])) -> (ndarray, [batch, 1]): potential function that takes the distance matrix and interaction parameters for each pair to compute the potential values.
        '''
        @wraps(func)
        def wrapper(*args):
            r'''
            Potential function created from interations.
            Args:
                args (tuple of ndarray,  [batch, N, 3] + Tuple([batch, Nterm])): the input arguments. The first one must be the position configuration of the atoms, the results are parameters for each interactions.
            Return:
                (ndarray, [batch, 1]): the potential value.
            '''
            vecs = args[0][:, idx]
            terms = termComputeFunc(vecs, eps)
            return torch.clip(func(terms, *args[1:]), max=1e8).sum(-1, keepdim=True)
        return wrapper
    return decorator