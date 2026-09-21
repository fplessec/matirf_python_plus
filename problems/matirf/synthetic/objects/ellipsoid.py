"""
Ellipsoid — a smooth, flat-lying blob: vesicles, endosomes, focal adhesions.

One kind, two biological readings depending on its sizes:

    vesicles / endosomes   lateral 100-400 nm, axial 80-200 nm, anywhere in depth
                           (the defaults) — point-like objects whose DEPTH must be recovered
    focal adhesions        lateral 1-5 um, axial 40-100 nm, 10-60 nm from the glass,
                           sharpness 2-5, tilt below 1 degree — flat plaques pressed
                           against the coverslip (a 4 um plaque tilted by 3 degrees would
                           already sink 100 nm at one end)

Why a Gaussian profile rather than a hard indicator: a hard edge only anti-aliases on the
grid and creates grid-aligned ringing that favours edge-preserving priors (TV) unfairly; a
smooth profile integrates cleanly on any grid, and is closer to a fluorophore cloud.
`sharpness` > 1 flattens the top ("soft binary") when a plateau is wanted.

Flat-slab pose: the two lateral sizes are drawn from the lateral range, the axial one from
its own (smaller) range, and the blob turns freely in the plane but tilts by at most
`max_tilt_deg` — it never pokes out of the shallow reconstructable depth.

Sizes are FULL visible sizes, about 4 standard deviations of the Gaussian.
"""

import torch

import settings as settings
from ..rng import uniform, random_flat_rotation
from .base import SyntheticObject, REACH, value_param, count_param


class Ellipsoid(SyntheticObject):
    """A rotated anisotropic 3D Gaussian: amplitude * exp(-0.5 (d^T M d)^sharpness)."""

    name = "Ellipsoid (vesicles, adhesions)"
    toml_key = "ellipsoid"
    ui_params = {
        "count": count_param("ellipsoids", default=20),
        "lateral_size_min_nm": value_param("Lateral size min", "L_{xy}^{min}", 150.0, "nm"),
        "lateral_size_max_nm": value_param("Lateral size max", "L_{xy}^{max}", 400.0, "nm"),
        "axial_size_min_nm": value_param("Axial size min", "L_z^{min}", 80.0, "nm"),
        "axial_size_max_nm": value_param("Axial size max", "L_z^{max}", 200.0, "nm"),
        "depth_min_nm": value_param("Centre depth min", "z^{min}", 30.0, "nm"),
        "depth_max_nm": value_param("Centre depth max", "z^{max}", 240.0, "nm"),
        "max_tilt_deg": value_param("Max out-of-plane tilt", "\\theta_{tilt}", 10.0, "deg"),
        "sharpness": value_param("Sharpness (1 = Gaussian)", "s", 1.0),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
    }

    def __init__(self, center_nm, sigmas_nm, rotation, amplitude=1.0, sharpness=1.0):
        self.center_nm, self.sigmas_nm = center_nm, sigmas_nm
        self.amplitude, self.sharpness = amplitude, sharpness
        s = torch.as_tensor(sigmas_nm, dtype=torch.float64)
        R = rotation.to(torch.float64)
        self._covariance = R @ torch.diag(s ** 2) @ R.T
        precision = R @ torch.diag(1.0 / s ** 2) @ R.T
        self._M = precision.to(device=settings.device, dtype=settings.dtype)
        self._c = torch.as_tensor(center_nm, device=settings.device, dtype=settings.dtype)

    @classmethod
    def sample(cls, p, gen, grid):
        ## the draw order below is part of the reproducibility contract: changing it
        ## changes every scene that contains ellipsoids
        lateral = uniform(gen, float(p["lateral_size_min_nm"]), float(p["lateral_size_max_nm"]), (2,))
        axial = uniform(gen, float(p["axial_size_min_nm"]), float(p["axial_size_max_nm"]))
        sigmas = (lateral[0].item() / 4.0, lateral[1].item() / 4.0, axial.item() / 4.0)
        margin = 0.08
        cx = uniform(gen, margin * grid.Lx_nm, (1 - margin) * grid.Lx_nm).item()
        cy = uniform(gen, margin * grid.Ly_nm, (1 - margin) * grid.Ly_nm).item()
        cz = uniform(gen, float(p["depth_min_nm"]), float(p["depth_max_nm"])).item()
        return cls(center_nm=(cx, cy, cz), sigmas_nm=sigmas,
                   rotation=random_flat_rotation(gen, float(p["max_tilt_deg"])),
                   amplitude=float(p["amplitude"]), sharpness=float(p["sharpness"]))

    def bounds(self):
        half = REACH * self._covariance.diagonal().sqrt()
        (cx, cy, cz), (hx, hy, hz) = self.center_nm, half.tolist()
        return (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)

    def density(self, X, Y, Z):
        dx, dy, dz = X - self._c[0], Y - self._c[1], Z - self._c[2]
        M = self._M
        q = (M[0, 0] * dx * dx + M[1, 1] * dy * dy + M[2, 2] * dz * dz
             + 2 * M[0, 1] * dx * dy + 2 * M[0, 2] * dx * dz + 2 * M[1, 2] * dy * dz)
        if self.sharpness != 1.0:
            q = q ** self.sharpness
        return self.amplitude * torch.exp(-0.5 * q)
