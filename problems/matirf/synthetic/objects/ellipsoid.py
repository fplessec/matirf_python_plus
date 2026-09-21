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

Where the centres go (`placement`):

    random   each centre at a random depth in [depth_min, depth_max] — independent objects
    plane    on an inclined plane crossing the volume: `surface_depth_nm` at the field
             centre, rising by `surface_rise_nm` across the field in a random direction —
             vesicles lined up along a slanted membrane, a depth GRADIENT to recover
    sphere   on a spherical cap of radius `surface_radius_nm`, deepest (`surface_depth_nm`)
             at the field centre and curving towards the glass — vesicles docked under a
             dome-shaped cell membrane

On a surface, each centre is scattered around it by `surface_jitter_nm` (std). Objects then
obey a spatial logic in depth, as they do in a cell, instead of floating independently.
"""

import math


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
        "depth_min_nm": {**value_param("Centre depth min", "z^{min}", 30.0, "nm"),
                         "depends_on": {"placement": "random"}},
        "depth_max_nm": {**value_param("Centre depth max", "z^{max}", 240.0, "nm"),
                         "depends_on": {"placement": "random"}},
        "max_tilt_deg": value_param("Max out-of-plane tilt", "\\theta_{tilt}", 10.0, "deg"),
        "sharpness": value_param("Sharpness (1 = Gaussian)", "s", 1.0),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
        "placement": {"title": "Placement of the centres", "type": "option",
                      "param_info": {"options_list": ["random", "plane", "sphere"]}},
        "surface_depth_nm": {**value_param("Surface depth at the centre", "z_s", 150.0, "nm"),
                             "depends_on": {"placement": ["plane", "sphere"]}},
        "surface_rise_nm": {**value_param("Plane: depth change across the field", "\\Delta z_s", 200.0, "nm"),
                            "depends_on": {"placement": "plane"}},
        "surface_radius_nm": {**value_param("Sphere: radius of curvature", "R_s", 10000.0, "nm"),
                              "depends_on": {"placement": "sphere"}},
        "surface_jitter_nm": {**value_param("Scatter around the surface", "\\sigma_s", 15.0, "nm"),
                              "depends_on": {"placement": ["plane", "sphere"]}},
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
    def build(cls, params, gen, grid):
        """For a plane, one random direction for the whole population, drawn first."""
        full = {**cls.default_params(), **(params or {})}
        if full["placement"] == "plane" and int(full.get("count", 0) or 0) > 0:
            full["_direction"] = uniform(gen, 0.0, 2 * math.pi).item()
        return super().build(full, gen, grid)

    @classmethod
    def surface_depth(cls, p, grid, x, y) -> float:
        """Depth of the placement surface at (x, y), in nm."""
        dx, dy = x - grid.Lx_nm / 2, y - grid.Ly_nm / 2
        if p["placement"] == "plane":
            field = math.hypot(grid.Lx_nm, grid.Ly_nm) / 2
            along = dx * math.cos(p["_direction"]) + dy * math.sin(p["_direction"])
            return float(p["surface_depth_nm"]) + float(p["surface_rise_nm"]) * along / (2 * field)
        radius, rho = float(p["surface_radius_nm"]), math.hypot(dx, dy)
        sag = radius - math.sqrt(radius * radius - rho * rho) if rho < radius else radius
        return float(p["surface_depth_nm"]) - sag

    @classmethod
    def sample(cls, p, gen, grid):
        ## the draw order below is part of the reproducibility contract: changing it
        ## changes every scene that contains ellipsoids (a surface only ADDS a final draw)
        lateral = uniform(gen, float(p["lateral_size_min_nm"]), float(p["lateral_size_max_nm"]), (2,))
        axial = uniform(gen, float(p["axial_size_min_nm"]), float(p["axial_size_max_nm"]))
        sigmas = (lateral[0].item() / 4.0, lateral[1].item() / 4.0, axial.item() / 4.0)
        margin = 0.08
        cx = uniform(gen, margin * grid.Lx_nm, (1 - margin) * grid.Lx_nm).item()
        cy = uniform(gen, margin * grid.Ly_nm, (1 - margin) * grid.Ly_nm).item()
        cz = uniform(gen, float(p["depth_min_nm"]), float(p["depth_max_nm"])).item()
        rotation = random_flat_rotation(gen, float(p["max_tilt_deg"]))
        if p["placement"] != "random":
            jitter = float(torch.randn((), generator=gen, dtype=torch.float64)) * float(p["surface_jitter_nm"])
            cz = min(max(cls.surface_depth(p, grid, cx, cy) + jitter, grid.z0_nm), grid.zN_nm)
        return cls(center_nm=(cx, cy, cz), sigmas_nm=sigmas, rotation=rotation,
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
