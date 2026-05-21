import torch
from torch.autograd.functional import jacobian

def forwardCartesian(points):
    """
    Forward transformation for point clouds.
    :param points: torch.Tensor of shape [batch, N, 3] where N > 4
    :return: (transformed_points: torch.Tensor [batch, N, 3], logdet: torch.Tensor [batch, 1])
    """
    B, N, _ = points.shape

    p1 = points[:, 0, :]  # [B, 3]
    p2 = points[:, 1, :]
    p3 = points[:, 2, :]
    rest = points[:, 3:, :]  # [B, N-3, 3]

    u = p2 - p1  # [B, 3]
    v = p3 - p1  # [B, 3]

    norm_u = torch.norm(u, dim=-1, keepdim=True)  # [B, 1]
    e_x = u / norm_u

    w = torch.cross(u, v, dim=-1)  # [B, 3]
    norm_w = torch.norm(w, dim=-1, keepdim=True)  # [B, 1]
    e_z = w / norm_w

    e_y = torch.cross(e_z, e_x, dim=-1)  # [B, 3]

    # Rotation matrix R with columns e_x, e_y, e_z
    R = torch.stack([e_x, e_y, e_z], dim=-1)  # [B, 3, 3]

    # Compute q_i = R^T (p_i - p1) for i >= 4
    diff = rest - p1[:, None, :]  # [B, N-3, 3]
    q = torch.bmm(R.transpose(1, 2), diff.transpose(1, 2)).transpose(1, 2)  # [B, N-3, 3]

    # Output tensor: first 3 points unchanged, rest are q
    transformed = torch.cat([points[:, :3, :], q], dim=1)  # [B, N, 3]

    # log abs det Jacobian is 0
    logdet = torch.zeros(B, 1).to(points)

    return transformed, logdet

def inverseCartesian(transformed):
    """
    Inverse transformation for point clouds.
    :param transformed: torch.Tensor of shape [batch, N, 3] where N > 4 (output of forward)
    :return: (original_points: torch.Tensor [batch, N, 3], logdet: torch.Tensor [batch, 1])
    """
    B, N, _ = transformed.shape

    p1 = transformed[:, 0, :]  # [B, 3]
    p2 = transformed[:, 1, :]
    p3 = transformed[:, 2, :]
    q = transformed[:, 3:, :]  # [B, N-3, 3]

    u = p2 - p1  # [B, 3]
    v = p3 - p1  # [B, 3]

    norm_u = torch.norm(u, dim=-1, keepdim=True)  # [B, 1]
    e_x = u / norm_u

    w = torch.cross(u, v, dim=-1)  # [B, 3]
    norm_w = torch.norm(w, dim=-1, keepdim=True)  # [B, 1]
    e_z = w / norm_w

    e_y = torch.cross(e_z, e_x, dim=-1)  # [B, 3]

    # Rotation matrix R with columns e_x, e_y, e_z
    R = torch.stack([e_x, e_y, e_z], dim=-1)  # [B, 3, 3]

    # Compute p_i = p1 + R q_i for i >= 4
    rest = p1[:, None, :] + torch.bmm(R, q.transpose(1, 2)).transpose(1, 2)  # [B, N-3, 3]

    # Output tensor: first 3 points unchanged, rest are reconstructed
    original = torch.cat([transformed[:, :3, :], rest], dim=1)  # [B, N, 3]

    # log abs det Jacobian is 0
    logdet = torch.zeros(B, 1).to(transformed)

    return original, logdet
