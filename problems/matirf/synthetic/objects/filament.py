"""
Filament — a thin fibre following a smooth curve: actin stress fibres, microtubules.

In an adherent cell these run roughly PARALLEL to the coverslip, at a height that changes
only slowly along them (a stress fibre rises from a focal adhesion near the glass into the
cell). So a filament here is:

    in the plane   a straight run of length L, bent by two smooth sine modes
                   (b1 sin(pi t) + b2 sin(2 pi t), zero at both ends): curved, never kinked
    in depth       a linear ramp from its start height, rising or sinking by at most
                   `axial_rise_nm` over its whole length

with a Gaussian RADIAL profile: `radius_nm` is its standard deviation, the visible width
(FWHM) is about 2.4 x radius. A fibre may leave the field of view, as real ones do.

Typical values: actin bundles and stress fibres radius 30-60 nm, length 5-20 um;
microtubules 10-15 nm. Keep in mind the slab is only ~300 nm deep: a radius of 60 nm already
makes a fibre 140 nm thick (FWHM), half the depth.

The density is the distance to the curve, sampled every radius/2 (at most 20 nm), which
bounds the sampling error to a few percent of the radius. It is evaluated only inside the
fibre's bounding box, and its lateral part once per lateral grid (see `_lateral`), so a
long fibre costs little.
"""

import math

import torch

import settings as settings
from ..rng import uniform, signed_uniform
from .base import SyntheticObject, REACH, value_param, count_param

## distance matrices are computed by chunks of this many points, to bound memory
_CHUNK = 16384


class Filament(SyntheticObject):
    """A smooth curve with a Gaussian radial profile: amplitude * exp(-0.5 (d / radius)^2)."""

    name = "Filament (stress fibres, microtubules)"
    toml_key = "filament"
    ui_params = {
        "count": count_param("filaments", default=8),
        "radius_nm": value_param("Radius (profile std)", "r", 30.0, "nm"),
        "length_min_nm": value_param("Length min", "\\ell^{min}", 3000.0, "nm"),
        "length_max_nm": value_param("Length max", "\\ell^{max}", 10000.0, "nm"),
        "bend": value_param("Bending (fraction of the length)", "\\beta", 0.1),
        "depth_min_nm": value_param("Start height min", "z^{min}", 60.0, "nm"),
        "depth_max_nm": value_param("Start height max", "z^{max}", 220.0, "nm"),
        "axial_rise_nm": value_param("Max rise along the fibre", "\\Delta z", 40.0, "nm"),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
    }

    def __init__(self, points_nm: torch.Tensor, radius_nm: float, amplitude: float = 1.0):
        self.radius_nm, self.amplitude = radius_nm, amplitude
        self._points = points_nm.to(device=settings.device, dtype=settings.dtype)
        low, high = points_nm.min(dim=0).values, points_nm.max(dim=0).values
        reach = REACH * radius_nm
        self._bounds = (float(low[0]) - reach, float(high[0]) + reach,
                        float(low[1]) - reach, float(high[1]) + reach,
                        float(low[2]) - reach, float(high[2]) + reach)

    @classmethod
    def sample(cls, p, gen, grid):
        radius = float(p["radius_nm"])
        length = uniform(gen, float(p["length_min_nm"]), float(p["length_max_nm"])).item()
        x0 = uniform(gen, 0.0, grid.Lx_nm).item()
        y0 = uniform(gen, 0.0, grid.Ly_nm).item()
        z_start = uniform(gen, float(p["depth_min_nm"]), float(p["depth_max_nm"])).item()
        rise = signed_uniform(gen, 0.0, float(p["axial_rise_nm"])) if float(p["axial_rise_nm"]) > 0 else 0.0
        theta = uniform(gen, 0.0, 2 * math.pi).item()
        bend = float(p["bend"]) * length
        b1 = signed_uniform(gen, 0.3, 1.0) * bend
        b2 = signed_uniform(gen, 0.0, 0.5) * bend

        spacing = min(20.0, radius / 2)
        t = torch.linspace(0.0, 1.0, max(16, math.ceil(length / spacing) + 1), dtype=torch.float64)
        along, across = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float64), \
            torch.tensor([-math.sin(theta), math.cos(theta)], dtype=torch.float64)
        offset = b1 * torch.sin(math.pi * t) + b2 * torch.sin(2 * math.pi * t)
        xy = (torch.tensor([x0, y0], dtype=torch.float64)
              + (t * length)[:, None] * along + offset[:, None] * across)
        z = z_start + t * rise
        return cls(torch.cat([xy, z[:, None]], dim=1), radius, float(p["amplitude"]))

    def bounds(self):
        return self._bounds

    def density(self, X, Y, Z):
        lateral, height = self._lateral(X, Y)
        squared = lateral * lateral + (Z - height) ** 2
        return self.amplitude * torch.exp(-0.5 * squared / self.radius_nm ** 2)

    def _lateral(self, X, Y):
        """
        For each (x, y): the lateral distance to the nearest curve point, and that point's z.

        The costly part — a distance to every curve point — depends only on (x, y), and the
        integrator evaluates the same lateral grid at every depth, so it is computed once
        and kept. Adding the vertical offset to the nearest point afterwards is exact for a
        horizontal fibre and a close upper bound for one rising by tens of nm over microns.
        """
        key = (X.data_ptr(), tuple(X.shape), Y.data_ptr())
        if getattr(self, "_key", None) != key:
            points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
            distance = torch.empty(points.shape[0], device=points.device, dtype=points.dtype)
            index = torch.empty(points.shape[0], device=points.device, dtype=torch.long)
            curve = self._points[:, :2]
            for start in range(0, points.shape[0], _CHUNK):
                nearest = torch.cdist(points[start:start + _CHUNK], curve).min(dim=1)
                distance[start:start + _CHUNK], index[start:start + _CHUNK] = nearest
            self._key = key
            self._cache = (distance.reshape(X.shape), self._points[index, 2].reshape(X.shape))
        return self._cache
