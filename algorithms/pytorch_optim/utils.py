import torch
import torch.nn.functional as F
from operations import apply_matirf_operator
from gui_dictionnaries.option_lists import REGULARIZATION_LIST


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_loss(f, g, H, reg, coeff, rho):
    """Calcule la fonction de perte selon la régularisation choisie"""
    zero = torch.zeros_like(f)
    norm_L1 = torch.nn.L1Loss(reduction='mean')
    norm_L2 = torch.nn.MSELoss(reduction='mean')
    data_term = norm_L2(apply_matirf_operator(H, f), g) / 2
    if reg == REGULARIZATION_LIST[0]:  # Sans régularisation
        loss = data_term
    elif reg == REGULARIZATION_LIST[1]:  # norme L1 de x : sparse
        loss = (1 - coeff) * data_term + coeff * norm_L1(f, zero)
    elif reg == REGULARIZATION_LIST[2]:  # norme L2 du grad : Tikhonov
        loss = (1 - coeff) * data_term + coeff * gradient_L2(f).mean()
    elif reg == REGULARIZATION_LIST[3]:  # norme L1 du grad : TV
        loss = (1 - coeff) * data_term + coeff * gradient_L1(f).mean()
    elif reg == REGULARIZATION_LIST[4]:  # norme de frobenius de la hessienne
        loss = (1 - coeff) * data_term + coeff * hessian_frobenius(f).mean()
    elif reg == REGULARIZATION_LIST[5]:  # shv regularization
        loss = (1 - coeff) * data_term + coeff * shv(f, weighting=rho).mean()
    elif reg == REGULARIZATION_LIST[6]:  # schatten norm d'ordre 1 de la hessienne
        loss = (1 - coeff) * data_term + coeff * hessian_schatten_1(f).mean()
    else:
        raise ValueError(f'Erreur: pas de régularisation associée à reg = {reg}')
    return loss

def spatial_grad(f):
    """Calcul du gradient spatial 3D de manière compatible avec autograd"""
    padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
    # Calcul des différences finies
    dz = padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]
    dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
    dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
    return dz, dy, dx

def gradient_L2(f, eps=1e-8):
    """Calcul de la norme L2 du gradient"""
    dz, dy, dx = spatial_grad(f)
    grad = dz.square() + dy.square() + dx.square()
    return torch.sqrt(grad + eps ** 2)

def gradient_L1(f):
    """Calcul de la norme L1 du gradient"""
    dz, dy, dx = spatial_grad(f)
    grad = torch.abs(dz) + torch.abs(dy) + torch.abs(dx)
    return grad

def hessian(f, delta=1.):
    """Calcul de la matrice hessienne"""
    # Calcul du gradient
    dz, dy, dx = spatial_grad(f)
    # Calcul des dérivées secondes avec autograd
    dxdz, dxdy, dxdx = spatial_grad(dx)
    dydz, dydy, dydx = spatial_grad(dy)
    dzdz, dzdy, dzdx = spatial_grad(dz)
    # Formation de la matrice hessienne
    hessian_components = [
        delta ** 2 * dzdz, delta * dzdy, delta * dzdx,
        delta * dydz, dydy, dydx,
        delta * dxdz, dxdy, dxdx
    ]
    # Reshape pour obtenir la matrice 3x3 pour chaque voxel
    hessian_matrix = torch.stack(hessian_components, dim=0)
    hessian_matrix = hessian_matrix.view(3, 3, f.shape[0], f.shape[1], f.shape[2])
    return hessian_matrix

def frobenius_norm(matrix, eps=1e-8):
    """Calcul de la norme de Frobenius"""
    # Utilisation de einsum pour la somme des carrés des éléments
    squared_sum = torch.einsum("ij...,ij...->...", matrix, matrix)
    return torch.sqrt(squared_sum + eps ** 2)

def schatten_norm_1(matrix):
    """Calcul de la norme de Schatten d'ordre 1 (somme des valeurs singulières)"""
    return torch.einsum("ii...", torch.abs(matrix))

def shv(f, delta=1., weighting=0.5, eps=1e-8):
    """Calcul de la régularisation SHV (Sparse Hessian Variation)"""
    hessian_matrix = hessian(f, delta)
    sparse_h_v = weighting * frobenius_norm(hessian_matrix, eps=eps) + (1 - weighting) * torch.abs(f)
    return sparse_h_v

def hessian_frobenius(f, delta=1., eps=1e-8):
    """Calcul de la norme de Frobenius de la hessienne"""
    hessian_matrix = hessian(f, delta)
    return frobenius_norm(hessian_matrix, eps=eps)

def hessian_schatten_1(f, delta=1.):
    """Calcul de la norme de Schatten d'ordre 1 de la hessienne"""
    hessian_matrix = hessian(f, delta)
    return schatten_norm_1(hessian_matrix)