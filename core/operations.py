"""
This file contains all the computational functions related to the physics of the multi-angle TIRF microscope.

However, it uses the fact that parameter sets are represented in this project by Python dictionaries.
Each parameter has a name (str) and a value, so a parameter set forms a dictionary.

To illustrate, the MA-TIRF operator requires two sets of parameters to be calculated:
    1. Measurement parameters:
        > "angles_deg": a list of the incident angle for each measurement stack, in degrees
        > "n_glass": the optical index of the incident medium
        > "n_medium": the optical index of the sample medium
        > "n_oil": the optical index of the immersion oil of the objective
        > "numerical_aperture": the numerical aperture of the objective
        > "wavelength_nm": the wavelength of the excitation light, in nanometers
        > "beam_divergence_deg": the divergence of the excitation beam, in degrees
    2. Operator parameters:
        > "nz": the number of cuts of the reconstructed image on the z axis
        > "z0": the smallest depth on z of the reconstructed image, in nanometers
        > "zN": the largest depth on z of the reconstructed image, in nanometers
        > "normalize": a boolean to choose whether to normalize the operator
The measurement parameters are already set for an MA-TIRF measurement stack, while the operator parameters are chosen
by the user for a given reconstruction.
"""


from math import pi
import torch
import numpy as np

import settings


def get_variables_from_dict(dico: dict, variable_name_list: list['str']) -> tuple:
    """
    This function allows you to unpack the values contained in a set of parameters represented by the ‘dico’ argument,
    using the ‘variable_name_list’ argument to select the keys (parameter names, therefore typed str) of the desired
    parameters from the set of parameters.
    It returns a tuple of the values of each desired parameter in the order determined by the ‘variable_name_list’.
    """
    return tuple(dico[key] for key in variable_name_list)

def compute_depths(z0: float, zN: float, nz: int) -> torch.Tensor:
    """
    Using 'z0' (the smallest depth on z of the reconstructed image, in nanometers) 'zN' (the smallest depth on z of the
    reconstructed image, in nanometers) and 'nz' (the number of cuts of the reconstructed image on the z axis), this
    function returns a 1D Tensor with a size nZ+1, which corresponds to the z axis that represents the desire
    reconstructed image.
    """
    return torch.linspace(z0, zN, nz + 1, device=settings.device, dtype=settings.dtype)

def compute_tirf_critical_angle(ni: float, nt: float) -> float:
    """
    Using 'ni' (the optical index of the incident medium) and 'nt' (the optical index of the sample medium), this
    function returns the critical angle of the microscope set up.
    """
    return np.arcsin(nt / ni) / pi * 180.  # in degrees

def compute_tirf_max_angle(NA: float, no: float) -> float:
    """
    Using 'NA' (the numerical aperture of the objective) and 'no' (the optical index of the immersion oil of the
    objective), this function returns the maximum angle of the microscope set up.
    """
    return np.arcsin(NA / no) / pi * 180. - 0.5  # in degrees

def compute_min_max_angles_from_params(measurement_params: dict) -> tuple[float, float]:
    """
    This function takes the measurement parameter dictionary 'measurement_params' and returns the minimum (critical
    angle) and maximum angles for this measurement.
    """
    theta_crit_deg = compute_tirf_critical_angle(measurement_params['n_glass'], measurement_params['n_medium'])
    theta_max_deg = compute_tirf_max_angle(measurement_params['numerical_aperture'], measurement_params['n_oil'])
    return theta_crit_deg, theta_max_deg


def compute_tirf_intensity_at_interface(angle_deg: float, ni: float, nt: float) -> float:
    """
    For a given incident angle 'angle_deg' in degrees, and using 'ni' (the optical index of the incident medium) and
    'nt' (the optical index of the sample medium), this function returns the value of the intensity of the transmitted
    beam at the interface (arbitrarily chosen at z=0+).
    """
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

def normalize_operator(A: torch.Tensor) -> torch.Tensor:
    """Given an operator 'A' (2D tensor), this function returns a normalized version of it."""
    U, S, V = torch.linalg.svd(A)
    return A / torch.max(S)

def compute_matirf_operator(angles_deg, nz, z0, zN, n_glass, n_medium, numerical_aperture, n_oil,
                            wavelength_nm, beam_divergence_deg, normalize, precision=settings.precision):
    """
    :param angles_deg: a python list of the incident angle for each measurement stack, in degrees
    :param nz: the number of cuts of the reconstructed image on the z axis
    :param z0: the smallest depth on z of the reconstructed image, in nanometers
    :param zN: the largest depth on z of the reconstructed image, in nanometers
    :param n_glass: the optical index of the incident medium
    :param n_medium: the optical index of the sample medium
    :param numerical_aperture: the numerical aperture of the objective
    :param n_oil: the optical index of the immersion oil of the objective
    :param wavelength_nm: the wavelength of the excitation light, in nanometers
    :param beam_divergence_deg: the divergence of the excitation beam, in degrees
    :param normalize: a boolean to choose whether to normalize the operator
    :param precision: used to approximate an integral by a sum of finite elements, the number of elements is 'precision'
    :return: a 2D torch.Tensor, which is the MA-TIRF operator of the given parameters
    """
    p = precision
    angles_deg = torch.tensor(angles_deg, device=settings.device, dtype=settings.dtype)
    n_angles = len(angles_deg)
    depths = compute_depths(z0, zN, nz)  # <-size nz+1
    theta_min_deg = compute_tirf_critical_angle(n_glass, n_medium)
    theta_max_deg = compute_tirf_max_angle(numerical_aperture, n_oil)
    ### Variables in capslock are 4D tensors for cleverly calculating integrals (except H: the operator).
    ### the 1st dim is each incident angle ie each MA-TIRF measurement stack ; integration variable: i
    ### the 2nd dim is each depth coordinate z of the desire reconstructed image ; integration variable: j
    ### the 3rd dim is a set of angles around each incident angle i ; integration variable: alpha
    ### the 4th dim is a layer of thicknesses along z corresponding to each coordinate j ; integration variable: z
    ### Those tensors will be used to integrate along alpha and z, in order to have a 2D torch.Tensor: the operator.
    ### The discretization of the alpha and z integrals is given by p = precision
    THETA = angles_deg.unsqueeze(1).unsqueeze(2).unsqueeze_(3).repeat(1, nz, p, p)  # shape(n_angles, nz, p, p)
    # THETA(i, ., ., .) invariant according to j, alpha, and z
    s = beam_divergence_deg / torch.cos(angles_deg / 180 * pi)
    alpha_ranges = (angles_deg[:, None] +
                    torch.linspace(-3, 3, p, device=settings.device) * s[:, None])  # shape(n_angles, p)
    ALPHA = alpha_ranges.unsqueeze(2).unsqueeze(3).repeat(1, 1, nz, p).permute(0, 2, 1, 3)  # shape(n_angles, nz, p, p)
    # ALPHA(i, ., alpha, .) invariant according to j and z
    S = s.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, nz, p, p)  # shape(n_angles, nz, p, p)
    # S(i, ., ., .) invariant according to j, alpha and z
    z_ranges = (depths[:-1].unsqueeze(1) + torch.linspace(0, 1, p, device=settings.device)
                * (depths[1:] - depths[:-1]).unsqueeze(1))  # shape(nz, P)
    Z = z_ranges.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_angles, p).permute(2, 0, 3, 1)  # shape(n_angles, nz, p, p)
    # Z(., j, ., z) invariant according to i and alpha
    ### Values outside the borders are cut off and replaced by the borders (0° and 89°):
    THETA[THETA < 0.] = 0.
    THETA[THETA > 89.] = 89.
    ALPHA[ALPHA < 0.] = 0.
    ALPHA[ALPHA > 89.] = 89.
    ### Then we can compute the all the physic
    if beam_divergence_deg == 0:  # considering the beam as a ray:
        WEIGHT = torch.zeros_like(ALPHA)
        WEIGHT[:, :, p // 2] = 1.
    else:  # considering the given divergence of the beam:
        WEIGHT = torch.exp(-0.5 * ((ALPHA - THETA) / S) ** 2)  # <-weights related to the Gaussian profile of the beam
    I0 = compute_tirf_intensity_at_interface(ALPHA, n_glass, n_medium)
    I0[ALPHA > theta_max_deg] = 0.  # <-condition: ray outside the aperture of the microscope lens
    KAPPA = torch.full_like(I0, 0., device=settings.device, dtype=settings.dtype)
    KAPPA[ALPHA > theta_min_deg] = 4 * pi / wavelength_nm * torch.sqrt(  # <-absorption coefficient (evanescent wave)
        (n_glass * torch.sin(ALPHA[ALPHA > theta_min_deg] / 180 * pi)) ** 2 - n_medium ** 2
    )  # ^condition: kappa=0 if reflection is not total (thus exp(-kappa*z = 1))
    I0_W_EXP_MINUS_KAPPA_Z = I0 * WEIGHT * torch.exp(- KAPPA * Z)
    # integration :
    H = I0_W_EXP_MINUS_KAPPA_Z.sum(dim=(2, 3))
    sw = WEIGHT.sum(dim=(2, 3))
    # normalisation :
    H = torch.where(sw > 1e-12, H / sw, torch.tensor(0))  # <-normalisation by the weights
    if normalize: H = normalize_operator(H)
    return H

def compute_matirf_operator_from_params(measurement_params: dict, operator_params: dict) -> torch.Tensor:
    """
    This function takes the measurement parameter dictionary 'measurement_params' and the operator parameter dictionary
    'operator_params' and returns the MA-TIRF operator of the given parameters.
    """
    (angles_deg, n_glass, n_medium, n_oil, numerical_aperture,
     wavelength_nm, beam_divergence_deg) = get_variables_from_dict(measurement_params, ['angles_deg',
                                                            'n_glass', 'n_medium', 'n_oil', 'numerical_aperture',
                                                            'wavelength_nm', 'beam_divergence_deg'])
    (nz, z0, zN, normalize) = get_variables_from_dict(operator_params, ['nz', 'z0', 'zN',
                                                                                        'normalize'])
    return compute_matirf_operator(angles_deg, nz, z0, zN, n_glass, n_medium, numerical_aperture, n_oil,
                                   wavelength_nm, beam_divergence_deg, normalize)

def apply_matirf_operator(H: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    This function computes the operation H*f where H is a MA-TIRF operator (2D torch.Tensor) and f a tensor from the
    mathematical set of the desired reconstructed image.
    This function can also be used to operate the other way: to compute the operation Ht*g where Ht is the conjugate
    transposed version of H and g is a tensor from the mathematical set of the MA-TIRF measurement stacks.
    """
    j = f.shape[0]
    f_flat = f.reshape(j, -1)        # (j, k*l)
    out = H @ f_flat                # (i, k*l)
    return out.reshape(H.shape[0], *f.shape[1:])
    #return torch.einsum('ij,jkl->ikl', H, f)

def estimate_delta_anisotropy(nz, z0, zN, nt, NA, wavelength_nm) -> float:
    """
    Using 'z0' (the smallest depth on z of the reconstructed image, in nanometers) 'zN' (the smallest depth on z of the
    reconstructed image, in nanometers) and 'nz' (the number of cuts of the reconstructed image on the z axis), we can
    compute Δz, the size of the voxel of the reconstructed image along z.
    Using 'nt' (the optical index of the sample medium), 'NA' (the numerical aperture of the objective) and
    'wavelength_nm' (the wavelength of the excitation beam in nm) we can estimate Δxy, the size of the the voxel of the
    reconstructed image along x or y.
    Under certain hypothesis, such as considering that the resolution is limited by diffraction, we can approximate:
    Δxy ≈ Δxy_Rayleigh = 0.61 λ / nt / NA
    This function return the estimated anisotropy ratio δ = Δz / Δxy.
    """
    # δ ≈ (zN - z0) / nz * n_medium * NA_obj / 0.61 / λ
    return  (zN - z0) / nz * nt * NA / 0.61 / wavelength_nm

def estimate_delta_anisotropy_from_params(measurement_params: dict, operator_params: dict) -> float:
    """
    This function takes the measurement parameter dictionary 'measurement_params' and the operator parameter dictionary
    'operator_params' and returns the estimated anisotropy ratio.
    """
    (n_medium, numerical_aperture, wavelength_nm) = get_variables_from_dict(measurement_params,
                                                ['n_medium', 'numerical_aperture', 'wavelength_nm'])
    (nz, z0, zN) = get_variables_from_dict(operator_params, ['nz', 'z0', 'zN'])
    return estimate_delta_anisotropy(nz, z0, zN, n_medium, numerical_aperture, wavelength_nm)
