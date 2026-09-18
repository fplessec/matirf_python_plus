"""
The optics of a multi-angle TIRF microscope — pure physics, no framework.

Nothing here imports from `core`, `solvers` or any GUI. Every function takes numbers and
returns numbers or tensors, so each one can be read, reasoned about and tested on its own.
`operator.py` wraps them into a ForwardOperator; this file never mentions that class.

--------------------------------------------------------------------------------------
The measurement, in one paragraph
--------------------------------------------------------------------------------------

In TIRF, a beam hits the glass/sample interface beyond the critical angle and is totally
reflected. What reaches the sample is an evanescent wave whose intensity decays exponentially
with depth z:

    I(z, theta) = I0(theta) * exp(-kappa(theta) * z)

The decay rate kappa grows with the incidence angle theta, so each angle probes a different
depth range: a shallow angle sees deep, a steep angle sees only the surface. Measuring at
several angles and inverting the resulting system recovers the axial distribution f(z) —
that is the whole idea of MA-TIRF, and H below is that system.

Row i of H corresponds to one incidence angle, column j to one depth slice, and H[i, j] is
how much slice j contributes to measurement i. Two integrals are evaluated numerically for
each entry: over the beam's angular spread (a real beam is not a single ray) and over the
thickness of the slice.

--------------------------------------------------------------------------------------
Parameters, and where they come from
--------------------------------------------------------------------------------------

    MEASUREMENT parameters — fixed by the instrument and the acquisition, read from the .json
        angles_deg           incidence angle of each measurement stack, in degrees
        n_glass              refractive index of the incident medium
        n_medium             refractive index of the sample medium
        n_oil                refractive index of the objective's immersion oil
        numerical_aperture   numerical aperture of the objective
        wavelength_nm        excitation wavelength, in nanometres
        beam_divergence_deg  angular spread of the beam, in degrees (0 = a perfect ray)

    OPERATOR parameters — chosen by the user for a given reconstruction
        nz                   number of z slices to reconstruct
        z0, zN               shallowest and deepest reconstructed depth, in nanometres
        normalize            divide H by its largest singular value
"""

from math import pi

import numpy as np
import torch

import settings as settings
import problems.matirf.settings as matirf_settings


# ── geometry: which angles are usable ─────────────────────────────────────────

def critical_angle(n_incident: float, n_sample: float) -> float:
    """
    Angle beyond which total internal reflection occurs, in degrees.

    Below it the beam refracts into the sample and there is no evanescent wave, so a stack
    acquired below this angle carries no depth information.
    """
    return np.arcsin(n_sample / n_incident) / pi * 180.0


def max_angle(numerical_aperture: float, n_oil: float) -> float:
    """
    Largest angle the objective can deliver, in degrees.

    The 0.5 degree margin keeps the very edge of the aperture out, where transmission is
    unreliable. A stack acquired ABOVE this angle carries no signal and is treated as a
    background measurement (see `split_background` in operator.py).
    """
    return np.arcsin(numerical_aperture / n_oil) / pi * 180.0 - 0.5


def angle_bounds(measurement_params: dict) -> tuple:
    """(critical, maximum) usable angles in degrees, from the measurement parameters."""
    return (critical_angle(measurement_params["n_glass"], measurement_params["n_medium"]),
            max_angle(measurement_params["numerical_aperture"], measurement_params["n_oil"]))


# ── the evanescent wave ───────────────────────────────────────────────────────

def depths(z0: float, zN: float, nz: int) -> torch.Tensor:
    """The nz+1 slice boundaries between z0 and zN — depths, not slice centres."""
    return torch.linspace(z0, zN, nz + 1, device=settings.device, dtype=settings.dtype)


def intensity_at_interface(angle_deg: torch.Tensor, n_incident: float,
                           n_sample: float) -> torch.Tensor:
    """
    Transmitted intensity just inside the sample (z = 0+), from the Fresnel coefficients.

    Averages the s and p polarizations as 0.75 * I_s + 0.25 * I_p, the weighting for the
    unpolarized-in-practice illumination used here. The +0j makes the square root complex,
    so it stays valid past the critical angle where the radicand turns negative — which is
    precisely the evanescent regime.
    """
    n = n_sample / n_incident
    angle_rad = angle_deg / 180.0 * pi
    c1 = torch.cos(angle_rad)
    c2 = torch.sqrt(n ** 2 - torch.sin(angle_rad) ** 2 + 0j)
    ts = 2.0 * c1 / (c1 + c2)
    tp = 2.0 * n * c1 / (n ** 2 * c1 + c2)
    return 0.75 * abs(ts * torch.conj(ts)) + 0.25 * abs(tp * torch.conj(tp))


def normalize_operator(H: torch.Tensor) -> torch.Tensor:
    """Divide H by its largest singular value, so ||H|| = 1."""
    return H / torch.linalg.svdvals(H).max()


def build_operator_matrix(angles_deg, nz, z0, zN, n_glass, n_medium, numerical_aperture,
                          n_oil, wavelength_nm, beam_divergence_deg, normalize,
                          precision: int = matirf_settings.precision) -> torch.Tensor:
    """
    The MA-TIRF matrix H, of shape (n_angles, nz).

    H[i, j] = average over the beam's angular spread around angle i, and over the thickness
    of slice j, of  I0(alpha) * exp(-kappa(alpha) * z),  weighted by the beam's Gaussian
    angular profile.

    Both integrals are evaluated as finite sums with `precision` samples, built as one
    4-D tensor (angle, slice, alpha, z) so the whole thing is a couple of vectorized
    operations rather than four nested loops:

        dim 0  i      one incidence angle = one measurement stack
        dim 1  j      one depth slice of the reconstruction
        dim 2  alpha  the angles sampled around incidence angle i
        dim 3  z      the depths sampled inside slice j
    """
    p = precision
    angles = torch.tensor(angles_deg, device=settings.device, dtype=settings.dtype)
    n_angles = len(angles)
    slice_edges = depths(z0, zN, nz)
    theta_min, theta_max = critical_angle(n_glass, n_medium), max_angle(numerical_aperture, n_oil)

    ## THETA[i,.,.,.] = the nominal incidence angle of stack i
    THETA = angles.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, nz, p, p)
    ## spread of the beam around each nominal angle (wider at grazing incidence):
    spread = beam_divergence_deg / torch.cos(angles / 180 * pi)
    alpha_ranges = (angles[:, None]
                    + torch.linspace(-3, 3, p, device=settings.device) * spread[:, None])
    ## ALPHA[i,.,alpha,.] = the sampled angles around stack i's nominal angle
    ALPHA = alpha_ranges.unsqueeze(2).unsqueeze(3).repeat(1, 1, nz, p).permute(0, 2, 1, 3)
    SPREAD = spread.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, nz, p, p)
    ## Z[.,j,.,z] = the depths sampled inside slice j
    z_ranges = (slice_edges[:-1].unsqueeze(1)
                + torch.linspace(0, 1, p, device=settings.device)
                * (slice_edges[1:] - slice_edges[:-1]).unsqueeze(1))
    Z = z_ranges.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_angles, p).permute(2, 0, 3, 1)

    ## angles outside [0, 89] are unphysical here; clamp rather than produce NaNs:
    THETA.clamp_(0.0, 89.0)
    ALPHA.clamp_(0.0, 89.0)

    if beam_divergence_deg == 0:
        ## a perfect ray: all the weight on the central sample
        WEIGHT = torch.zeros_like(ALPHA)
        WEIGHT[:, :, p // 2] = 1.0
    else:
        WEIGHT = torch.exp(-0.5 * ((ALPHA - THETA) / SPREAD) ** 2)   # Gaussian beam profile

    I0 = intensity_at_interface(ALPHA, n_glass, n_medium)
    I0[ALPHA > theta_max] = 0.0          # outside the objective's aperture: no light

    ## decay rate of the evanescent wave; zero below the critical angle, where the beam
    ## refracts instead of reflecting and exp(-kappa z) must stay 1:
    KAPPA = torch.zeros_like(I0)
    evanescent = ALPHA > theta_min
    KAPPA[evanescent] = 4 * pi / wavelength_nm * torch.sqrt(
        (n_glass * torch.sin(ALPHA[evanescent] / 180 * pi)) ** 2 - n_medium ** 2)

    integrand = I0 * WEIGHT * torch.exp(-KAPPA * Z)
    H = integrand.sum(dim=(2, 3))
    weight_sum = WEIGHT.sum(dim=(2, 3))
    ## divide by the total weight to make each row an average, not a sum:
    H = torch.where(weight_sum > 1e-12, H / weight_sum, torch.tensor(0.0))
    return normalize_operator(H) if normalize else H


def estimate_anisotropy_ratio(nz, z0, zN, n_medium, numerical_aperture,
                              wavelength_nm) -> float:
    """
    delta = dz / dxy, the ratio of axial to lateral voxel size.

    Axial: dz = (zN - z0) / nz, directly from the reconstruction grid.
    Lateral: taken as the diffraction limit sampled at Nyquist,
        dxy_Rayleigh = 0.61 * lambda / (n * NA),   dxy = dxy_Rayleigh / 2 = 0.305 * lambda / (n * NA)

    Regularizers that weight the axial derivative need this ratio; it is what the GUI's
    "Estimate" button next to `delta` computes.
    """
    return (zN - z0) / nz * n_medium * numerical_aperture / 0.305 / wavelength_nm
