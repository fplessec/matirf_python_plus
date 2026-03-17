import torch
import torch.nn.functional as F
from operations import apply_matirf_operator
from gui_dictionnaries.option_lists import REGULARIZATION_LIST

"""
δ = Δz / Δxy << 1
delta est le ratio de la taille des pixels sur la direction z par rapport à celle sur les directions xy.
Si on veut régulariser le gradient spacial de l'image 3D f super-résolue sur z, il faut prendre en compte l'anisotropie !
> || grad(f) ||² = dz² + dy² + dx² 
  ici on considère une variation entre deux pixels sur z aussi importante qu'une variation entre deux pixels sur x/y
  or entre deux pixels sur x/y on peut mettre parfois 10 à 30 pixels sur z
  ainsi en regularisant on regularise de la meme manière sur xy que sur z donc on sur-régularise la direction z 
  (sur-lissage etc)
  autrement dit, la direction z étant plus fine elle est beaucoup plus sensible à la régularisation
> || grad(f) ||² = Δz² * dz² +  Δxy² * dy² + Δxy² * dx²  ou bien:
  || grad(f) ||² = δ² * dz² +  dy² + dx² 
  ici en prenant en compte l'anisotropie on corrige la sur-régularisation sur z qui peut  etre responsable de la 
  destruction de la super-résolution
  
Comment calculer delta ?
Δz = (zN - z0) / nz  est choisi par l'utilisateur
Δxy = FOV_camera / Npixel_xy
Si FOV_camera est inconnu, on peut estimer Δxy à partir des paramètres de mesures sous l'hypothèse suivante:
la résolution est limitée par la diffraction et donc la taille du pixel est bien choisie pour echantillonnée:
Δxy ≈ Δxy_Rayleigh / 2  (Nyquist)
et :
Δxy_Rayleigh ≈ 0.61 λ / n_medium / NA_obj  qui sont dans les paramètres de mesure (indice optique, longueur d'onde, 
ouverture numérique, ...)
Comme on veut juste un ordre de grandeur pour delta on va ignoré Nyquist et prendre Δxy ≈ Δxy_Rayleigh

On a donc:
δ ≈ (zN - z0) / nz * n_medium * NA_obj / 0.61 / λ
"""



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_loss(f, g, H, reg, lambda_reg, rho, delta=1.):
    """Calcule la fonction de perte selon la régularisation choisie"""
    zero = torch.zeros_like(f)
    norm_L1 = torch.nn.L1Loss(reduction='mean')
    norm_L2 = torch.nn.MSELoss(reduction='mean')
    data_term = norm_L2(apply_matirf_operator(H, f), g) / 2
    if reg == REGULARIZATION_LIST[0]:  # Sans régularisation
        loss = data_term
    elif reg == REGULARIZATION_LIST[1]:  # norme L1 de x : sparse
        loss = (1 - lambda_reg) * data_term + lambda_reg * norm_L1(f, zero)
    elif reg == REGULARIZATION_LIST[2]:  # norme L2 du grad : Tikhonov
        loss = (1 - lambda_reg) * data_term + lambda_reg * gradient_L2(f, delta=delta).mean()
    elif reg == REGULARIZATION_LIST[3]:  # norme L1 du grad : TV
        loss = (1 - lambda_reg) * data_term + lambda_reg * gradient_L1(f, delta=delta).mean()
    elif reg == REGULARIZATION_LIST[4]:  # norme de frobenius de la hessienne
        loss = (1 - lambda_reg) * data_term + lambda_reg * hessian_frobenius(f, delta=delta).mean()
    elif reg == REGULARIZATION_LIST[5]:  # shv regularization
        loss = (1 - lambda_reg) * data_term + lambda_reg * shv(f, weighting=rho, delta=delta).mean()
    elif reg == REGULARIZATION_LIST[6]:  # schatten norm d'ordre 1 de la hessienne
        loss = (1 - lambda_reg) * data_term + lambda_reg * hessian_schatten_1(f, delta=delta).mean()
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

def gradient_L2(f, delta=1., eps=1e-8):
    """Calcul de la norme L2 du gradient"""
    dz, dy, dx = spatial_grad(f)
    grad = delta**2 * dz.square() + dy.square() + dx.square()
    return torch.sqrt(grad + eps ** 2)

def gradient_L1(f, delta=1.):
    """Calcul de la norme L1 du gradient"""
    dz, dy, dx = spatial_grad(f)
    grad = delta * torch.abs(dz) + torch.abs(dy) + torch.abs(dx)
    return grad

def hessian(f, delta=1.):
    """Computation of the 3D Hessian matrix."""
    # gradient computation:
    dz, dy, dx = spatial_grad(f)
    # second partial derivatives computation:
    dxdz, dxdy, dxdx = spatial_grad(dx)
    dydz, dydy, dydx = spatial_grad(dy)
    dzdz, dzdy, dzdx = spatial_grad(dz)
    # formation of the Hessian matrix:
    hess = torch.stack([
        delta**2 * dzdz, delta * dzdy, delta * dzdx,
        delta * dydz,    dydy,         dydx,
        delta * dxdz,    dxdy,         dxdx
    ])
    return hess.view(3, 3, *f.shape)


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