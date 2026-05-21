import torch
from torch.autograd.functional import jacobian

def compute_torsion(pN, pA, pB, pX):
    ab = pA - pN
    bc = pB - pA
    cd = pX - pB
    t = torch.cross(ab, bc, dim=-1)
    u = torch.cross(bc, cd, dim=-1)

    norm_t = torch.norm(t, dim=-1, keepdim=True)
    norm_u = torch.norm(u, dim=-1, keepdim=True)
    cos_chi = torch.sum(t * u, dim=-1, keepdim=True) / (norm_t * norm_u + 1e-8)
    cross_tu = torch.cross(t, u, dim=-1)
    sin_chi = - torch.sum(bc * cross_tu, dim=-1, keepdim=True) / (torch.norm(bc, dim=-1, keepdim=True) * norm_t * norm_u + 1e-8)
    chi = torch.atan2(sin_chi, cos_chi)
    return chi

def forwardTorsion(pos):
    pN = pos[:,0]
    pA = pos[:,1]
    pB = pos[:,2]
    pX = pos[:,3]
    vec_BX = pX - pB
    r = torch.norm(vec_BX, dim=-1, keepdim=True) + 1e-8
    vec_BA = pA - pB
    l = torch.norm(vec_BA, dim=-1, keepdim=True) + 1e-8
    cos_theta = torch.sum(vec_BA * vec_BX, dim=-1, keepdim=True) / (l * r)
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    chi = compute_torsion(pN, pA, pB, pX)
    transformed = pos.clone()
    transformed[:,3] = torch.cat([r, theta, chi], dim=-1)
    logdet = -2 * torch.log(r) - torch.log(torch.sin(theta) + 1e-8)
    return transformed, logdet

def inverseTorsion(trans):
    pN = trans[:,0]
    pA = trans[:,1]
    pB = trans[:,2]
    r = trans[:,3,0:1]
    theta = trans[:,3,1:2]
    chi = trans[:,3,2:3]
    vec_BA = pA - pB
    l = torch.norm(vec_BA, dim=-1, keepdim=True) + 1e-8
    z = vec_BA / l
    v_ref = pN - pA
    dot = torch.sum(v_ref * z, dim=-1, keepdim=True)
    perp = v_ref - dot * z
    d = torch.norm(perp, dim=-1, keepdim=True) + 1e-8
    x = perp / d
    y = torch.cross(z, x, dim=-1)
    dir_vec = (torch.cos(theta) * z + torch.sin(theta) * (torch.cos(chi) * x + torch.sin(chi) * y))
    pX = pB + r * dir_vec
    recon = trans.clone()
    recon[:,3] = pX
    logdet = 2 * torch.log(r) + torch.log(torch.sin(theta) + 1e-8)
    return recon, logdet
