from math import pi
import torch
import numpy as np

from settings import device



def get_variables_from_dict(dico, variable_name_list):
    return tuple(dico[key] for key in variable_name_list)




def compute_depths(z0: float, zN: float, nz: int) -> torch.Tensor:
    return torch.linspace(z0, zN, nz + 1, device=device)


def compute_tirf_critical_angle(ni: float, nt: float) -> float:
    return np.arcsin(nt / ni) / pi * 180.  # in degrees


def compute_tirf_max_angle(NA: float, no: float) -> float:
    return np.arcsin(NA / no) / pi * 180. - 0.5  # in degrees


def compute_min_max_angles_from_params(params):
    theta_crit_deg = compute_tirf_critical_angle(params['n_glass'], params['n_medium'])
    theta_max_deg = compute_tirf_max_angle(params['numerical_aperture'], params['n_oil'])
    return theta_crit_deg, theta_max_deg


def compute_tirf_intensity_at_interface(angle_deg, ni, nt):
    n = nt / ni
    angle_rad = angle_deg / 180. * pi
    delta2 = n ** 2 - torch.sin(angle_rad) ** 2 + 0J
    c1 = torch.cos(angle_rad)
    c2 = torch.sqrt(delta2)
    ts = 2. * c1 / (c1 + c2)
    tp = 2. * n * c1 / (n ** 2 * c1 + c2)
    Ip = tp * torch.conj(tp)
    Is = ts * torch.conj(ts)
    return 0.75 * abs(Is) + 0.25 * abs(Ip)


def normalize_operator(A):
    # return A / A.pow(2).sum().sqrt()
    # return (A - A.min()) / (A.max() - A.min())
    U, S, V = torch.linalg.svd(A)
    return A / torch.max(S)


def compute_matirf_operator(angles_deg, depths, nz, n_glass, n_medium, numerical_aperture, n_oil,
                            wavelength_nm, beam_divergence_deg,
                            normalize,precision=100):
    angles_deg = torch.tensor(angles_deg, device=device, dtype=torch.float32)
    n_angles = len(angles_deg)
    P = precision + 1
    theta_min_deg = compute_tirf_critical_angle(n_glass, n_medium)
    theta_max_deg = compute_tirf_max_angle(numerical_aperture, n_oil)
    ###
    THETA = angles_deg.unsqueeze(1).unsqueeze(2).unsqueeze_(3).repeat(1, nz, P, P)  # shape(n_angles, nz, P, P)
    # THETA(i, ., ., .) invariant selon j, alpha et z
    s = beam_divergence_deg / torch.cos(angles_deg / 180 * pi)
    alpha_ranges = angles_deg[:, None] + torch.linspace(-3, 3, P, device=device) * s[:, None]  # shape(n_angles, P)
    ALPHA = alpha_ranges.unsqueeze(2).unsqueeze(3).repeat(1, 1, nz, P).permute(0, 2, 1, 3)  # shape(n_angles, nz, P, P)
    # ALPHA(i, ., alpha, .) invariant selon j et z
    S = s.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, nz, P, P)  # shape(n_angles, nz, P, P)
    # S(i, ., ., .) invariant selon j, alpha et z
    z_ranges = (depths[:-1].unsqueeze(1) + torch.linspace(0, 1, P, device=device)
                * (depths[1:] - depths[:-1]).unsqueeze(1))  # shape(nz, P)
    Z = z_ranges.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_angles, P).permute(2, 0, 3, 1)  # shape(n_angles, nz, P, P)
    # Z(., j, ., z) invariant selon i et alpha
    ###
    THETA[THETA < 0.] = 0.
    THETA[THETA > 89.] = 89.
    ALPHA[ALPHA < 0.] = 0.
    ALPHA[ALPHA > 89.] = 89.
    ###
    if beam_divergence_deg == 0:
        WEIGHT = torch.zeros_like(ALPHA)
        WEIGHT[:, :, P // 2] = 1.
    else:
        WEIGHT = torch.exp(-0.5 * ((ALPHA - THETA) / S) ** 2)  # <-poids liés au profile gaussien du faisceau
    I0 = compute_tirf_intensity_at_interface(ALPHA, n_glass, n_medium)
    I0[ALPHA > theta_max_deg] = 0.  # <-condition: rayon hors de l'ouverture de l'objectif de microscope
    KAPPA = torch.full_like(I0, 0., device=device)
    KAPPA[ALPHA > theta_min_deg] = 4 * pi / wavelength_nm * torch.sqrt(  # <-coefficient d'absorption (onde evanescente)
        (n_glass * torch.sin(ALPHA[ALPHA > theta_min_deg] / 180 * pi)) ** 2 - n_medium ** 2
    )  # ^condition: kappa=0 si réflexion non totale (ainsi exp(-kappa*z = 1))
    I0_W_EXP_MINUS_KAPPA_Z = I0 * WEIGHT * torch.exp(- KAPPA * Z)
    ###
    # intégration :
    H = I0_W_EXP_MINUS_KAPPA_Z.sum(dim=(2, 3))
    sw = WEIGHT.sum(dim=(2, 3))
    # normalisation :
    H = torch.where(sw > 1e-12, H / sw, torch.tensor(0))
    if normalize: H = normalize_operator(H)
    return H




def compute_matirf_operator_from_params(matirf_params, operator_params):
    (angles_deg, n_glass, n_medium, n_oil, numerical_aperture,
     wavelength_nm, beam_divergence_deg) = get_variables_from_dict(matirf_params, ['angles_deg',
                                                            'n_glass', 'n_medium', 'n_oil', 'numerical_aperture',
                                                            'wavelength_nm', 'beam_divergence_deg'])
    (nz, z0, zN, normalize) = get_variables_from_dict(operator_params, ['nz', 'z0','zN',
                                                                                        'normalize'])
    depths = compute_depths(z0, zN, nz)
    return compute_matirf_operator(angles_deg, depths, nz, n_glass, n_medium,
                                   numerical_aperture, n_oil,
                                   wavelength_nm, beam_divergence_deg,
                                   normalize)

def apply_matirf_operator(H, f):
    return torch.einsum('ij,jkl->ikl', H, f)

