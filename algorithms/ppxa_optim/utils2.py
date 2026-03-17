import torch
import torch.nn.functional as F



def prox_regularization(x, reg, lam, rho=0.5):

    if reg == "L1":
        return prox_l1(x, lam)

    elif reg == "Tikhonov":
        return prox_K_norm(x, K_grad, KT_grad, lam, p=2)

    elif reg == "TV":
        return prox_K_norm(x, K_grad, KT_grad, lam, p=1)

    elif reg == "Hessian Frobenius":
        return prox_K_norm(x, K_hessian, KT_hessian, lam, p=2)

    elif reg == "Hessian Schatten":
        return prox_K_norm(x, K_hessian, KT_hessian, lam, p=1)

    elif reg == "SHV":
        return prox_shv(x, lam, rho)

    else:
        return x


# ============================================================
# OPERATEURS DIFFERENTIELS
# ============================================================


def spatial_grad(f):
    """Computation of the 3D spatial gradient in a way it's compatible with autograd."""
    padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
    # finite difference computation: (first order approximation of the gradient)
    dz = padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]
    dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
    dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
    return dz, dy, dx

def divergence(dz, dy, dx):
    """Computation of the 3D spatial divergence from the gradient: adjoint operator of the gradient."""
    div = torch.zeros_like(dz)
    div[1:-1,:,:] += dz[1:-1,:,:] - dz[:-2,:,:]
    div[:,1:-1,:] += dy[:,1:-1,:] - dy[:,:-2,:]
    div[:,:,1:-1] += dx[:,:,1:-1] - dx[:,:,:-2]
    return div


def hessian(f):
    """Hessienne 3D"""

    dz, dy, dx = spatial_grad(f)

    dxdz, dxdy, dxdx = spatial_grad(dx)
    dydz, dydy, dydx = spatial_grad(dy)
    dzdz, dzdy, dzdx = spatial_grad(dz)

    H = torch.stack([
        dzdz, dzdy, dzdx,
        dydz, dydy, dydx,
        dxdz, dxdy, dxdx
    ])

    return H.view(3,3,*f.shape)


# ============================================================
# PROX OPERATORS
# ============================================================

def prox_l1(x, lam):
    """Soft thresholding"""
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0)


# ============================================================
# PROX GENERIQUE ||Kx||_p
# primal-dual
# ============================================================

def prox_K_norm(x, K, KT, lam, p=1, n_iter=20):

    u = x.clone()

    if p == 1:
        y = torch.zeros_like(K(x))
    else:
        y = torch.zeros_like(K(x))

    tau = 0.5
    sigma = 0.5

    for _ in range(n_iter):

        Ku = K(u)

        if p == 1:
            y = torch.clamp(y + sigma * Ku, -lam, lam)

        elif p == 2:
            y = (y + sigma * Ku) / (1 + sigma * lam)

        u = (x - tau * KT(y)) / (1 + tau)

    return u


# ============================================================
# OPERATEURS K
# ============================================================

def K_identity(x):
    return x


def KT_identity(x):
    return x


def K_grad(x):
    dz, dy, dx = spatial_grad(x)
    return torch.stack((dz, dy, dx))


def KT_grad(p):
    dz, dy, dx = p
    return divergence(dz, dy, dx)


def K_hessian(x):
    return hessian(x)


def KT_hessian(H):
    return torch.sum(H, dim=(0,1))


# ============================================================
# SHV
# ============================================================

def prox_shv(x, lam, rho=0.5, step=0.1, n_iter=10):

    u = x.clone()

    for _ in range(n_iter):

        H = hessian(u)

        grad_h = torch.sum(H, dim=(0,1))

        grad_l1 = torch.sign(u)

        grad = rho * grad_h + (1-rho) * grad_l1

        u = u - step * lam * grad

    return u
