import torch
import torch.nn.functional as F

from operations import apply_matirf_operator


# ============================================================
# DIFFERENTIAL OPERATORS
# ============================================================

def spatial_grad(f, delta=1.):
    """
    Compute approximated 3D spatial gradient using finite differences.
    delta applies an anisotropy factor along the z axis.
    """
    padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
    # finite difference computation: (first order approximation of the gradient)
    dz = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) * delta
    dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
    dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
    return dz, dy, dx

def divergence(z, y, x):
    """Compute the 3D divergence (adjoint of gradient) of a vector field (z, y, x) using finite differences."""
    div = torch.zeros_like(z)
    # z direction:
    div[1:-1, :, :] += z[1:-1, :, :] - z[:-2, :, :]
    div[0, :, :] += z[0, :, :]
    div[-1, :, :] -= z[-2, :, :]
    # y direction:
    div[:, 1:-1, :] += y[:, 1:-1, :] - y[:, :-2, :]
    div[:, 0, :] += y[:, 0, :]
    div[:, -1, :] -= y[:, -2, :]
    # x direction:
    div[:, :, 1:-1] += x[:, :, 1:-1] - x[:, :, :-2]
    div[:, :, 0] += x[:, :, 0]
    div[:, :, -1] -= x[:, :, -2]
    return div

def laplacian(f, delta=1.):
    """Compute approximate 3D Laplacian using finite differences."""
    dz, dy, dx = spatial_grad(f, delta=delta)
    return divergence(dz, dy, dx)

def hessian(f, delta=1.):
    """Computation of the 3D Hessian matrix."""
    dz, dy, dx = spatial_grad(f, delta=delta)
    # second partial derivatives:
    dxdz, dxdy, dxdx = spatial_grad(dx, delta=delta)
    dydz, dydy, dydx = spatial_grad(dy, delta=delta)
    dzdz, dzdy, dzdx = spatial_grad(dz, delta=delta)
    # formation of the Hessian matrix:
    hess = torch.stack([
        dzdz, dzdy, dzdx,
        dydz, dydy, dydx,
        dxdz, dxdy, dxdx
    ])
    return hess.view(3, 3, *f.shape)

# ============================================================
# LOSS (DEPENDING ON REGULARIZATION PENALTY)
# ============================================================

def compute_loss(f, g, H, reg, lambda_reg, rho, delta=1.):
    """Calcule la fonction de perte selon la régularisation choisie"""
    zero = torch.zeros_like(f)
    norm_L1 = torch.nn.L1Loss(reduction='mean')
    norm_L2 = torch.nn.MSELoss(reduction='mean')
    data_term = 0.5 * norm_L2(apply_matirf_operator(H, f), g)
    positivity_term = distance_positivity(f)
    positivity_term = 0
    if reg == "no regularization":
        return data_term + positivity_term
    elif reg == "L1":
        return data_term + positivity_term + lambda_reg * norm_L1(f, zero)
    elif reg == "Tikhonov":
        return data_term + positivity_term + lambda_reg * gradient_L2(f, delta=delta).mean()
    elif reg == "Tikhonov Boulanger":
        return data_term + positivity_term + lambda_reg * gradient_L2(f, delta=delta).mean()
    elif reg == "TV":
        return data_term + positivity_term + lambda_reg * gradient_L1(f, delta=delta).mean()
    elif reg == "TV normalized":
        return data_term + positivity_term + lambda_reg * gradient_L1(f, delta=delta).mean()
    elif reg == "Hessian Frobenius":
        return data_term + positivity_term + lambda_reg * hessian_frobenius(f, delta=delta).mean()
    elif reg == "Hessian Schatten":
        return data_term + positivity_term + lambda_reg * hessian_schatten_1(f, delta=delta).mean()
    elif reg == "SHV":
        return data_term + positivity_term + lambda_reg * shv(f, rho=rho, delta=delta).mean()
    else:
        raise ValueError(f"Unknown regularization method '{reg}'.")

def distance_positivity(f):
    neg_vals = torch.minimum(f, torch.tensor(0., device=f.device))
    return 0.5 * torch.mean(neg_vals**2)

def gradient_L2(f, delta=1., eps=1e-8):
    """Calcul de la norme L2 du gradient"""
    dz, dy, dx = spatial_grad(f, delta=delta)
    grad = dz.square() + dy.square() + dx.square()
    return torch.sqrt(grad + eps ** 2)

def gradient_L1(f, delta=1.):
    """Calcul de la norme L1 du gradient"""
    dz, dy, dx = spatial_grad(f, delta=delta)
    grad = torch.abs(dz) + torch.abs(dy) + torch.abs(dx)
    return grad

def frobenius_norm(matrix, eps=1e-8):
    """Calcul de la norme de Frobenius"""
    # Utilisation de einsum pour la somme des carrés des éléments
    squared_sum = torch.einsum("ij...,ij...->...", matrix, matrix)
    return torch.sqrt(squared_sum + eps ** 2)

def schatten_norm_1(matrix):
    """Calcul de la norme de Schatten d'ordre 1 (somme des valeurs singulières)"""
    return torch.einsum("ii...", torch.abs(matrix))

def hessian_frobenius(f, delta=1., eps=1e-8):
    """Calcul de la norme de Frobenius de la hessienne"""
    hessian_matrix = hessian(f, delta)
    return frobenius_norm(hessian_matrix, eps=eps)

def hessian_schatten_1(f, delta=1.):
    """Calcul de la norme de Schatten d'ordre 1 de la hessienne"""
    hessian_matrix = hessian(f, delta)
    return schatten_norm_1(hessian_matrix)

def shv(f, delta=1., rho=0.6, eps=1e-8):
    """Calcul de la régularisation SHV (Sparse Hessian Variation)"""
    hessian_matrix = hessian(f, delta)
    sparse_h_v = rho * frobenius_norm(hessian_matrix, eps=eps) + (1 - rho) * torch.abs(f)
    return sparse_h_v


# ============================================================
# CONSTRAINT AND REGULARIZATIONS
# ============================================================

def prox_positivity(u):
    #"""Prox de la contrainte de positivité"""
    #return 0.5 * (u + torch.maximum(u, torch.tensor(0., device=u.device)))

    """Contrainte dure de positivité"""
    return torch.clamp(u, min=0)  #NEW

def prox_regularization(f, reg, lambda_reg, delta=1., rho=0.6):
    if reg == "no regularization":
        return f
    elif reg == "L1":
        return prox_l1(f, lambda_reg)
    elif reg == "Tikhonov":
        return prox_tikhonov(f, lambda_reg, delta=delta)
    elif reg == "Tikhonov Boulanger":
        return prox_tikhonov_boulanger(f, lambda_reg, delta=delta)
    elif reg == "TV":
        return prox_tv(f, lambda_reg, delta=delta)
    elif reg == "TV normalized":
        return prox_tv_normalized(f, lambda_reg, delta=delta)
    elif reg == "Hessian Frobenius":
        return prox_hessian_frobenius(f, lambda_reg, delta=delta)
    elif reg == "Hessian Schatten":
        return prox_hessian_schatten(f, lambda_reg, delta=delta)
    elif reg == "SHV":
        return prox_shv(f, lambda_reg, delta=delta, rho=rho)
    else:
        raise ValueError(f"Unknown regularization method '{reg}'.")

# ============================================================
# PROX REGULARISATIONS
# ============================================================

def prox_l1(f, lambda_reg):
    """Soft-thresholding proximal for L1 norm."""
    return torch.sign(f) * torch.clamp(torch.abs(f) - lambda_reg, min=0)

def prox_tikhonov(f, lambda_reg, delta=1., tau=None, n_iter=50):
    """
    Proximal operator of L2 gradient norm (anisotropic Tikhonov).
    Args:
        f: input 3D image [Z,Y,X]
        lambda_reg: regularization coefficient
        delta: anisotropy ratio between the z axi and the xy axis
        tau: step size for gradient descent ; default = 1 / (1 + 12*lambda_reg)
        n_iter: number of iterations
    """
    if tau is None:
        tau = 1/(1+12*lambda_reg)
    u = f.clone()
    for _ in range(n_iter):
        lap = laplacian(u, delta=delta)
        grad = u - f - 2*lambda_reg*lap
        u = u - tau*grad
    return u

def prox_tikhonov_boulanger(f, lambda_reg, delta=1., dt=10., max_iter=50):
    """
    Proximal operator of anisotropic Tikhonov regularization (faithful to Boulanger's PDE).
    Minimizes: ||u - x||^2 + lambda_reg * ||grad(u)||^2_aniso
    Args:
        f: input 3D image [Z,Y,X]
        lambda_reg: regularization weight
        delta: anisotropy ratio Δz/Δxy
        dt: base time step
        max_iter: number of iterations
    """
    u = f.clone()
    for _ in range(max_iter):
        lap = laplacian(u, delta=delta)
        velocity = f - u + lambda_reg * lap
        v_min, v_max = velocity.min(), velocity.max()
        u = u + dt / (v_max - v_min + 1e-8) * velocity  # adaptive step
    return u

def prox_tv(f, lambda_reg, delta=1., tau=0.125, n_iter=20):
    """
    Proximal Operator of the L1 norm of the gradient (Total Variation regularization) using Chambolle’s method, with
    anisotropy delta along z-axis.
    Args:
        f: input 3D image [Z,Y,X]
        lambda_reg: regularization coefficient
        delta: anisotropy ratio between the z axi and the xy axis
        tau: dual step size (<= 1/6 for stability in 3D) ; default 1/8
        n_iter: number of Chambolle iterations (empirical 20–50)
    """
    # initialize dual variables:
    pz = torch.zeros_like(f)
    py = torch.zeros_like(f)
    px = torch.zeros_like(f)
    for _ in range(n_iter):
        # update the primal variable:
        u = f - lambda_reg * divergence(pz, py, px)
        # update the dual variables (Chambolle step):
        dz, dy, dx = spatial_grad(u, delta=delta)
        denom = 1.0 + tau * torch.sqrt(dz**2 + dy**2 + dx**2 + 1e-12)
        pz = (pz + tau * dz) / denom
        py = (py + tau * dy) / denom
        px = (px + tau * dx) / denom
    return f - lambda_reg * divergence(pz, py, px)

def prox_tv_normalized(f, lambda_reg, delta=1.0, tau=0.125, n_iter=50):
    """
    Total Variation denoising (proximal Chambolle) in the style of Boulanger.
    Args:
        f (Tensor): input 3D image [Z,Y,X]
        lambda_reg (float): regularization weight
        delta (float): anisotropy ratio Δz/Δxy
        tau (float): dual step size
        n_iter (int): number of Chambolle iterations
    """
    # initialize dual variables:
    pz = torch.zeros_like(f)
    py = torch.zeros_like(f)
    px = torch.zeros_like(f)
    for _ in range(n_iter):
        # update the primal variable:
        u = f - lambda_reg * divergence(pz, py, px)
        # update the dual variables (Chambolle step):
        dz, dy, dx = spatial_grad(u, delta=delta)
        pz_new = pz + tau * dz
        py_new = py + tau * dy
        px_new = px + tau * dx
        # normalization to constrain ||p|| ≤ 1:
        norm = torch.maximum(torch.ones_like(pz), torch.sqrt(pz_new**2 + py_new**2 + px_new**2))
        pz = pz_new / norm
        py = py_new / norm
        px = px_new / norm
    return f - lambda_reg * divergence(pz, py, px)

def prox_hessian_frobenius(f, lambda_reg, delta= 1., step=0.1, n_iter=10):
    """
    Proximal of Frobenius norm of Hessian using gradient descent.

    step: gradient step size
    n_iter: number of iterations
    """
    u = f.clone()
    for _ in range(n_iter):
        hess = hessian(u, delta=delta)
        grad = torch.sum(hess, dim=(0,1))
        u = u - step * lambda_reg * grad
    return u

def prox_hessian_schatten(f, lambda_reg, delta=1., step=0.1, n_iter=10):
    """
    Proximal of Schatten-1 norm of Hessian (approx).

    step: gradient step size
    n_iter: number of iterations
    """
    u = f.clone()
    for _ in range(n_iter):
        hess = hessian(u, delta=delta)
        grad = torch.sign(hess).sum(dim=(0,1))
        u = u - step * lambda_reg * grad
    return u

def prox_shv(f, lambda_reg, delta=1., rho=0.5, step=0.1, n_iter=10):
    """
    Proximal operator of the Sparse Hessian Variation (SHV).

    rho: weight between Hessian and L1 term
    step: gradient step
    n_iter: number of iterations
    """
    u = f.clone()
    for _ in range(n_iter):
        hess = hessian(u, delta=delta)
        grad_h = torch.sum(hess, dim=(0,1))
        grad_l1 = torch.sign(u)
        grad = rho * grad_h + (1-rho) * grad_l1
        u = u - step * lambda_reg * grad
    return u


