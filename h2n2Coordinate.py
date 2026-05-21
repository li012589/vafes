import numpy as np
import torch
import matplotlib.pyplot as plt

from forceUtils.energy import energy
from forceUtils.twobody import fourthPowerBond, coulombPair
from forceUtils.threebody import harmonicAngle, harmonicCosine
from forceUtils.fourbody import periodicProperDihedral


def rotMatFromVec(vector):
    assert vector.shape == (3,), "Input must be a 3-dimensional vector"
    assert torch.isclose(vector[2], torch.tensor(0.0)), "Z component must be zero"

    x, y = vector[0], vector[1]

    # Compute the rotation angle around the z-axis.
    theta = -torch.atan2(y, x)

    # Build the rotation matrix.
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0.0],
        [sin_theta, cos_theta, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=vector.dtype)
    return rotation_matrix


def _cv2Coord(config):
    r'''
    convert from Cv space to coord space, config shoud be (x1, y1, d, x2, y2, z2).
    1. (x1 - d, y1, 0) is the coord of H1;
    2. (-d, 0, 0) is the coord of N1;
    3. (0, 0, 0) is the coord of N2;
    4. (x2, y2, z2) is the coord of H2.
    '''
    pos = torch.zeros(config.shape[0], 4, 3).to(config)
    # H1
    pos[:, 0, 0] = config[:, 0] - config[:, 2]
    pos[:, 0, 1] = config[:, 1]
    # N1
    pos[:, 1, 0] = -config[:, 2]
    #H2
    pos[:, 3, :] = config[:, 3:]
    return pos


def _coord2Cv(pos):
    r'''
    convert from coord space to cv space, cv space is (x1, y1, d, x2, y2, z2). coord order is (H1, N1, N2, H2)
    '''
    origin = pos[:, 2:3, :]  # use the third point as the origin
    points_translated = pos - origin  # translated points

    # Use the translated second point (index 1) as the negative x direction.
    x_neg = points_translated[:, 1, :]  # shape (batch_size, 3)

    # Compute the normalized positive x direction.
    x_dir = -x_neg / torch.norm(x_neg, dim=1, keepdim=True)  # positive x direction

    # Use the first point (index 0) to define the xy-plane.
    v = points_translated[:, 0, :]  # shape (batch_size, 3)

    # Decompose v into parallel and perpendicular parts to the x-axis.
    v_dot_x = torch.sum(v * x_dir, dim=1, keepdim=True)
    v_parallel = v_dot_x * x_dir
    v_perpendicular = v - v_parallel

    # Compute the normalized y direction.
    y_dir = v_perpendicular / torch.norm(v_perpendicular, dim=1, keepdim=True)

    # Compute the z direction to enforce a right-handed frame.
    z_dir = torch.cross(x_dir, y_dir, dim=1)

    # Build the rotation matrix with x, y, z as columns.
    rotation_matrix = torch.stack([x_dir, y_dir, z_dir], dim=-1)

    # Apply the rotation matrix to the translated points.
    _pos = torch.bmm(points_translated, rotation_matrix)

    cv = torch.cat([_pos[:, 0, :2] - _pos[:, 1, :2], -_pos[:, 1, :1], _pos[:, 3, :]], dim=-1)
    return cv


def energyCV(config, mass, charge, functs, idxs, params):
    r'''
    config shoud be (x1, y1, d, x2, y2, z2).
    1. (x1 - d, y1, 0) is the coord of H1;
    2. (-d, 0, 0) is the coord of N1;
    3. (0, 0, 0) is the coord of N2;
    4. (x2, y2, z2) is the coord of H2.
    '''
    pos = _cv2Coord(config)
    return energy(pos, mass, charge, functs, idxs, params)
