import torch

from forceUtils.utils import dihedralFromPos


def alanineDipeptidePhiPsi(data: torch.Tensor):
    data = data.reshape(-1, 13, 3)
    pos = data[:, [[1, 3, 5, 8], [3, 5, 8, 10]]]
    dihedrals = dihedralFromPos(pos)
    return dihedrals[:, 0], dihedrals[:, 1]
