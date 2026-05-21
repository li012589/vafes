import torch


def compute_frame_cylindrical(P):
    """
    Compute the local cylindrical frame from the first three points.
    P: [batch, N, 3]
    Returns: O [batch, 3], R [batch, 3, 3] (rotation matrix)
    """
    batch = P.shape[0]
    O = P[:, 0, :]  # Origin
    v_z = P[:, 1, :] - O
    ez = v_z / torch.norm(v_z, dim=-1, keepdim=True)

    v = P[:, 2, :] - O
    v_perp = v - torch.einsum('bi,bi->b', v, ez).unsqueeze(-1) * ez  # Keep einsum here as it's scalar-dot, efficient
    ex = v_perp / torch.norm(v_perp, dim=-1, keepdim=True)

    ey = torch.cross(ez, ex, dim=-1)
    ey = ey / torch.norm(ey, dim=-1, keepdim=True)  # Normalize for safety

    R = torch.stack([ex, ey, ez], dim=-1)  # [batch, 3, 3]

    det = torch.det(R)
    assert torch.allclose(det, torch.ones_like(det)), "Rotation matrix det not 1"

    return O, R

def compute_frame(points_abc):
    """
    points_abc: (batch_size, 3, 3) tensor
    Returns: A (bs,3), s1 (bs,), proj (bs,), s (bs,), theta (bs,), R (bs,3,3)
    """
    bs = points_abc.shape[0]
    A = points_abc[:, 0]
    B = points_abc[:, 1]
    C = points_abc[:, 2]

    vec_AB = B - A
    s1 = torch.norm(vec_AB, dim=1)
    u_hat = vec_AB / torch.clip(s1.unsqueeze(1), min=1e-10)

    vec_AC = C - A
    proj = torch.sum(vec_AC * u_hat, dim=1)

    perp_sq = torch.norm(vec_AC, dim=1)**2 - proj**2
    perp_sq = torch.clamp(perp_sq, min=0.0)
    s = torch.sqrt(perp_sq)

    perp = vec_AC - proj.unsqueeze(1) * u_hat
    y_hat = perp / torch.clip(s.unsqueeze(1), min=1e-10)

    z_hat = torch.cross(u_hat, y_hat, dim=1)

    R = torch.stack([u_hat, y_hat, z_hat], dim=2)

    return A, s1, proj, s, R

def euler_from_rotation(R):
    """
    R: (bs,3,3)
    Returns: alpha (bs,), beta (bs,), gamma (bs,), cos_beta (bs,)
    """
    R31 = R[:, 2, 0]
    beta = torch.arcsin(-R31)
    cos_beta = torch.sqrt(torch.clip(1 - R31**2, min=1e-10))
    alpha = torch.atan2(R[:, 1, 0] / cos_beta, R[:, 0, 0] / cos_beta)
    gamma = torch.atan2(R[:, 2, 1] / cos_beta, R[:, 2, 2] / cos_beta)
    return alpha, beta, gamma, cos_beta

def rotation_from_euler(alpha, beta, gamma):
    """
    alpha, beta, gamma: (bs,)
    Returns: R (bs,3,3)
    """
    bs = alpha.shape[0]
    device = alpha.device

    cos_a = torch.cos(alpha).unsqueeze(1).unsqueeze(1)
    sin_a = torch.sin(alpha).unsqueeze(1).unsqueeze(1)
    Rz = torch.cat([
        torch.cat([cos_a, -sin_a, torch.zeros(bs, 1, 1, device=device)], dim=2),
        torch.cat([sin_a, cos_a, torch.zeros(bs, 1, 1, device=device)], dim=2),
        torch.cat([torch.zeros(bs, 1, 1, device=device), torch.zeros(bs, 1, 1, device=device), torch.ones(bs, 1, 1, device=device)], dim=2)
    ], dim=1)

    cos_b = torch.cos(beta).unsqueeze(1).unsqueeze(1)
    sin_b = torch.sin(beta).unsqueeze(1).unsqueeze(1)
    Ry = torch.cat([
        torch.cat([cos_b, torch.zeros(bs, 1, 1, device=device), sin_b], dim=2),
        torch.cat([torch.zeros(bs, 1, 1, device=device), torch.ones(bs, 1, 1, device=device), torch.zeros(bs, 1, 1, device=device)], dim=2),
        torch.cat([-sin_b, torch.zeros(bs, 1, 1, device=device), cos_b], dim=2)
    ], dim=1)

    cos_g = torch.cos(gamma).unsqueeze(1).unsqueeze(1)
    sin_g = torch.sin(gamma).unsqueeze(1).unsqueeze(1)
    Rx = torch.cat([
        torch.cat([torch.ones(bs, 1, 1, device=device), torch.zeros(bs, 1, 1, device=device), torch.zeros(bs, 1, 1, device=device)], dim=2),
        torch.cat([torch.zeros(bs, 1, 1, device=device), cos_g, -sin_g], dim=2),
        torch.cat([torch.zeros(bs, 1, 1, device=device), sin_g, cos_g], dim=2)
    ], dim=1)

    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    return R

def cartesian2cylindrical(xyz):
    """
    Input: xyz [..., 3]
    Output: cyl [..., 3], logdet [...,]
            cyl = (r, theta, z'), logdet = log|detJ|
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    r = torch.sqrt(x**2 + z**2)
    theta = torch.atan2(z, x)
    zp = y

    cyl = torch.stack([r, theta, zp], dim=-1)
    # log|det J| = -log(r)
    logdet = -torch.log(r + 1e-20)  # avoid r=0 issues
    return cyl, logdet

def cylindrical2cartesian(cyl):
    """
    Input: cyl [..., 3]
    Output: xyz [..., 3], logdet [...,]
            xyz = (x,y,z), logdet = log|detJ|
    """
    r = cyl[..., 0]
    theta = cyl[..., 1]
    zp = cyl[..., 2]

    x = r * torch.cos(theta)
    y = zp
    z = r * torch.sin(theta)

    xyz = torch.stack([x, y, z], dim=-1)
    # log|det J| = log(r)
    logdet = torch.log(r + 1e-20)
    return xyz, logdet

def forwardCylindrical(P, eps=1e-6):
    """
    Forward bijection: P -> Q (vectorized, no loop)
    Q[:, :3, :] = P[:, :3, :]
    For i >= 3: Q[:, i, :] = [rho, theta, z] in local frame
    Returns: Q [batch, N, 3], log_det [batch]
    """
    batch, N, _ = P.shape

    O, R = compute_frame_cylindrical(P)

    # Process all remaining points at once
    M = N - 3
    if M == 0:
        return P.clone(), torch.zeros(batch, device=P.device)

    remaining_points = P[:, 3:, :]  # [batch, M, 3]
    centered = remaining_points - O.unsqueeze(1)  # [batch, M, 3]
    # Local coords: R^T @ centered (batched matmul)
    local = torch.matmul(R.transpose(-2, -1), centered.transpose(-2, -1)).transpose(-2, -1)  # [batch, M, 3]

    lx = local[..., 0]
    ly = local[..., 1]
    lz = local[..., 2]

    rho = torch.sqrt(lx**2 + ly**2 + eps)
    theta = torch.atan2(ly, lx)
    z = lz

    remaining_Q = torch.stack([rho, theta, z], dim=-1)  # [batch, M, 3]
    Q = torch.cat([P[:, :3, :], remaining_Q], dim=1)  # [batch, N, 3]

    log_det = -torch.log(rho).sum(dim=-1)  # [batch]

    return Q, log_det

def inverseCylindrical(Q, eps=1e-6):
    batch, N, _ = Q.shape

    O, R = compute_frame_cylindrical(Q[:, :3, :])  # Frame from first three points in Q (same as P)

    # Process all remaining points at once
    M = N - 3

    remaining_Q = Q[:, 3:, :]  # [batch, M, 3]
    rho = remaining_Q[..., 0]
    theta = remaining_Q[..., 1]
    z = remaining_Q[..., 2]

    lx = rho * torch.cos(theta)
    ly = rho * torch.sin(theta)
    lz = z

    local = torch.stack([lx, ly, lz], dim=-1)  # [batch, M, 3]
    # Global coords: O + R @ local (batched matmul)
    remaining_P = O.unsqueeze(1) + torch.matmul(R, local.transpose(-2, -1)).transpose(-2, -1)  # [batch, M, 3]

    P = torch.cat([Q[:, :3, :], remaining_P], dim=1)  # [batch, N, 3]

    log_det_inv = torch.log(rho + eps).sum(dim=-1)  # [batch]

    return P, log_det_inv

def forwardPolar(points):
    """
    points: (batch_size, N, 3) tensor
    Returns: params (batch_size, 9): [xa,ya,za, alpha,beta,gamma, s1,s2,theta], logdet (batch_size)
    """
    bs = points.shape[0]

    A, s1, proj, s, R = compute_frame(points[:, :3])
    extra_points = points[:, 3:]

    theta = torch.atan2(s, proj)
    s2 = torch.sqrt(torch.clip(proj**2 + s**2, min=1e-10))
    alpha, beta, gamma, cos_beta = euler_from_rotation(R)

    extra_delta = extra_points - A.unsqueeze(1)
    extra_local = torch.bmm(extra_delta, R)

    params = torch.cat([A, alpha.unsqueeze(1), beta.unsqueeze(1), gamma.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(1), theta.unsqueeze(1), extra_local.view(bs, -1)], dim=1)
    logdet = -2 * torch.log(torch.clip(s1, min=1e-10)) - 2 * torch.log(torch.clip(s2, min=1e-10)) - torch.log(torch.clip(torch.abs(torch.sin(theta)), min=1e-10)) - torch.log(torch.clip(cos_beta, min=1e-10))
    return params, logdet

def inversePolar(params):
    """
    params: (batch_size,3*N): [xa,ya,za, alpha,beta,gamma, s1,s2,theta]
    Returns: points (bs,N,3), logdet (bs)
    """
    bs = params.shape[0]
    N = params.shape[1] // 3
    M = N - 3

    A = params[:, :3]
    alpha = params[:, 3]
    beta = params[:, 4]
    gamma = params[:, 5]
    s1 = params[:, 6]
    s2 = params[:, 7]
    theta = params[:, 8]
    extra_local_flat = params[:, 9:]
    extra_local = extra_local_flat.view(bs, M, 3)

    R = rotation_from_euler(alpha, beta, gamma)

    vec_B = s1.unsqueeze(1) * R[:, :, 0]
    B = A + vec_B

    local_C = torch.stack([s2 * torch.cos(theta), s2 * torch.sin(theta), torch.zeros_like(theta)], dim=1)
    vec_C = torch.bmm(R, local_C.unsqueeze(2)).squeeze(2)
    C = A + vec_C

    extra_delta = torch.bmm(extra_local, R.transpose(1, 2))
    P_extra = A.unsqueeze(1) + extra_delta

    points = torch.cat([A.unsqueeze(1), B.unsqueeze(1), C.unsqueeze(1), P_extra], dim=1)

    cos_b = torch.cos(beta)
    logdet = 2 * torch.log(torch.clip(s1, min=1e-10)) + 2 * torch.log(torch.clip(s2, min=1e-10)) + torch.log(torch.clip(torch.abs(torch.sin(theta)), min=1e-10)) + torch.log(torch.clip(torch.abs(cos_b), min=1e-10))

    return points, logdet

def forward(points):
    """
    points: (batch_size, N, 3) tensor, N > 3, first 3 points define the frame
    Returns: outputs (batch_size, N*3): [O_x, O_y, O_z, alpha, beta, gamma, r, proj, s, then flat local coords of points 4 to N]
             log_det_J (batch_size,)
    """
    bs = points.shape[0]

    points_abc = points[:, :3, :]
    extra_points = points[:, 3:, :]

    A, s1, proj, s, R = compute_frame(points_abc)

    alpha, beta, gamma, cos_beta = euler_from_rotation(R)

    extra_delta = extra_points - A.unsqueeze(1)
    extra_local = torch.bmm(extra_delta, R)

    outputs = torch.cat([
        A, alpha.unsqueeze(1), beta.unsqueeze(1), gamma.unsqueeze(1),
        s1.unsqueeze(1), proj.unsqueeze(1), s.unsqueeze(1),
        extra_local.view(bs, -1)
    ], dim=1)

    log_det_J = -2 * torch.log(torch.clip(s1, min=1e-10)) - torch.log(torch.clip(s, min=1e-10)) - torch.log(torch.clip(cos_beta, min=1e-10))

    return outputs, log_det_J

def inverse(outputs):
    """
    outputs: (batch_size, 3*N): [O_x, O_y, O_z, alpha, beta, gamma, r, proj, s, then flat local coords of points 4 to N]
    Returns: points (batch_size, N, 3)
             log_det_J_inv (batch_size,)
    """
    bs = outputs.shape[0]
    N = outputs.shape[1] // 3
    M = N - 3

    A = outputs[:, :3]
    alpha = outputs[:, 3]
    beta = outputs[:, 4]
    gamma = outputs[:, 5]
    r = outputs[:, 6]
    proj = outputs[:, 7]
    s = outputs[:, 8]
    extra_local_flat = outputs[:, 9:]
    extra_local = extra_local_flat.view(bs, M, 3)

    R = rotation_from_euler(alpha, beta, gamma)

    P1 = A.unsqueeze(1)
    P2 = A.unsqueeze(1) + r.unsqueeze(1).unsqueeze(2) * R[:, :, 0].unsqueeze(1)
    P3 = A.unsqueeze(1) + proj.unsqueeze(1).unsqueeze(2) * R[:, :, 0].unsqueeze(1) + s.unsqueeze(1).unsqueeze(2) * R[:, :, 1].unsqueeze(1)

    extra_delta = torch.bmm(extra_local, R.transpose(1, 2))
    P_extra = A.unsqueeze(1) + extra_delta

    points = torch.cat([P1, P2, P3, P_extra], dim=1)

    cos_beta = torch.cos(beta)
    log_det_J_forward = -2 * torch.log(r + 1e-10) - torch.log(s + 1e-10) - torch.log(torch.abs(cos_beta) + 1e-10)
    log_det_J_inv = - log_det_J_forward

    return points, log_det_J_inv
