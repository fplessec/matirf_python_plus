"""
Ellipsoid — a flat, in-plane-oriented Gaussian blob (bead / vesicle / node).

Why a Gaussian profile rather than a binary indicator (1 inside / 0 outside):
    > a hard edge only anti-aliases at the boundary and creates grid-aligned ringing
      that unfairly favours edge-preserving regularizers (TV) and pollutes comparisons ;
    > a Gaussian is smooth, differentiable, integrates cleanly onto any grid, and its
      "characteristic size" is simply the standard deviation ;
    > it is closer to reality (a fluorophore cloud / PSF-limited structure is smooth).
A quasi-binary flat-topped variant is available via the 'sharpness' parameter (>1).

Flat-slab constraint: the two lateral half-widths are drawn from a lateral size range,
the axial one from a SEPARATE (smaller) axial range, and the blob is only rotated in
the plane (about z) with a small optional tilt — so it never pokes out of the shallow
(<= ~500 nm) reconstructable depth.
"""

from dataclasses import dataclass

import torch

import common.settings as settings
from ..rng import uniform, random_flat_rotation
from .base import ContinuousObject, SyntheticObjectType, value_param, count_param


@dataclass
class GaussianEllipsoid(ContinuousObject):
    """
    A rotated anisotropic 3D Gaussian.

    center_nm : (cx, cy, cz) centre in nm.
    sigmas_nm : (s1, s2, s3) standard deviations along the principal axes (s1, s2 lateral,
                s3 axial). Visible full size along an axis ~ 4·sigma.
    rotation  : 3x3 tensor orienting the principal axes (use random_flat_rotation).
    amplitude : peak value before global [0, 1] normalization.
    sharpness : radial-profile shape; 1 = Gaussian, >1 = flat-topped ("soft binary").

    Density: amplitude · exp(-0.5 · (dᵀ M d)^sharpness), M = R diag(1/s²) Rᵀ.
    """

    center_nm: tuple
    sigmas_nm: tuple
    rotation: torch.Tensor
    amplitude: float = 1.0
    sharpness: float = 1.0

    def __post_init__(self):
        s = torch.as_tensor(self.sigmas_nm, dtype=torch.float64)
        R = self.rotation.to(torch.float64)
        precision = R @ torch.diag(1.0 / s ** 2) @ R.T
        self._M = precision.to(device=settings.device, dtype=settings.dtype)
        self._c = torch.as_tensor(self.center_nm, device=settings.device, dtype=settings.dtype)

    def density(self, X, Y, Z):
        dx, dy, dz = X - self._c[0], Y - self._c[1], Z - self._c[2]
        M = self._M
        q = (M[0, 0] * dx * dx + M[1, 1] * dy * dy + M[2, 2] * dz * dz
             + 2 * M[0, 1] * dx * dy + 2 * M[0, 2] * dx * dz + 2 * M[1, 2] * dy * dz)
        if self.sharpness != 1.0:
            q = q ** self.sharpness
        return self.amplitude * torch.exp(-0.5 * q)


class Ellipsoid(SyntheticObjectType):
    """Flat, in-plane-oriented Gaussian ovaloids."""

    name = "Ellipsoid"
    toml_key = "ellipsoid"

    ui_params = {
        "count": count_param("ellipsoids", default=8),
        "lateral_size_min_nm": value_param("Lateral size min", "L_{xy}^{min}", 200.0, "nm"),
        "lateral_size_max_nm": value_param("Lateral size max", "L_{xy}^{max}", 1500.0, "nm"),
        "axial_size_min_nm": value_param("Axial size min", "L_z^{min}", 40.0, "nm"),
        "axial_size_max_nm": value_param("Axial size max", "L_z^{max}", 150.0, "nm"),
        "max_tilt_deg": value_param("Max out-of-plane tilt", "\\theta_{tilt}", 10.0, "deg"),
        "sharpness": value_param("Sharpness (1=Gaussian)", "s", 1.0),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
        "depth_min_frac": value_param("Centre depth min (frac)", "z^{min}", 0.0),
        "depth_max_frac": value_param("Centre depth max (frac)", "z^{max}", 0.8),
    }

    @classmethod
    def sample(cls, params, gen, grid) -> GaussianEllipsoid:
        p = params
        lat = uniform(gen, float(p["lateral_size_min_nm"]), float(p["lateral_size_max_nm"]), (2,))
        ax = uniform(gen, float(p["axial_size_min_nm"]), float(p["axial_size_max_nm"]))
        sigmas = (lat[0].item() / 4.0, lat[1].item() / 4.0, ax.item() / 4.0)
        margin = 0.08
        cx = uniform(gen, margin * grid.Lx_nm, (1 - margin) * grid.Lx_nm).item()
        cy = uniform(gen, margin * grid.Ly_nm, (1 - margin) * grid.Ly_nm).item()
        span = grid.zN_nm - grid.z0_nm
        cz = uniform(gen, grid.z0_nm + float(p["depth_min_frac"]) * span,
                     grid.z0_nm + float(p["depth_max_frac"]) * span).item()
        return GaussianEllipsoid(
            center_nm=(cx, cy, cz), sigmas_nm=sigmas,
            rotation=random_flat_rotation(gen, float(p["max_tilt_deg"])),
            amplitude=float(p["amplitude"]), sharpness=float(p["sharpness"]),
        )
