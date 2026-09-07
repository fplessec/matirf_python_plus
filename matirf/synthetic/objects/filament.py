"""
Filament — a thin tube following a smooth 3D curve (actin / microtubule proxy).

The tube has a Gaussian RADIAL profile of standard deviation 'radius_nm' around a
poly-line. To respect the flat slab the curve meanders mostly in the (x, y) plane and
varies only slightly in z (bounded axial wobble). Each filament spans a random segment
of the field; the segment is bent sideways in-plane so fibres look curved, not straight.
"""

from dataclasses import dataclass

import torch

import common.settings as settings
from ..rng import uniform, signed_uniform
from .base import ContinuousObject, SyntheticObjectType, value_param, count_param


@dataclass
class Filament(ContinuousObject):
    """
    points_nm : (K, 3) points along the curve, spaced finer than 'radius_nm'.
    radius_nm : radial half-width of the fibre.
    amplitude : peak value before normalization.

    Density: amplitude · exp(-0.5 · (d_min / radius_nm)²), d_min = distance to the curve.
    """

    points_nm: torch.Tensor
    radius_nm: float
    amplitude: float = 1.0

    def __post_init__(self):
        self._pts = torch.as_tensor(self.points_nm, device=settings.device, dtype=settings.dtype)

    def density(self, X, Y, Z):
        shape = X.shape
        P = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)  # (N, 3)
        out = torch.empty(P.shape[0], device=P.device, dtype=P.dtype)
        chunk = 8192  # bound the (N, K) distance matrix
        for i in range(0, P.shape[0], chunk):
            d = torch.cdist(P[i:i + chunk], self._pts)  # (b, K) euclidean distances
            out[i:i + chunk] = d.min(dim=1).values
        return self.amplitude * torch.exp(-0.5 * (out / self.radius_nm) ** 2).reshape(shape)


def _curve_points(gen, p0, p1, sag_frac, axial_wobble_nm, n_ctrl=3) -> torch.Tensor:
    """Smooth poly-line from p0 to p1 (nm), bent sideways in-plane, small axial wobble."""
    p0 = torch.as_tensor(p0, dtype=torch.float64)
    p1 = torch.as_tensor(p1, dtype=torch.float64)
    length = torch.linalg.norm(p1 - p0).item()
    dirxy = (p1 - p0)[:2]
    perp = torch.tensor([-dirxy[1], dirxy[0], 0.0], dtype=torch.float64)
    n = torch.linalg.norm(perp)
    if n > 0:
        perp = perp / n
    ctrl = [p0]
    for i in range(1, n_ctrl + 1):
        t = i / (n_ctrl + 1)
        base = (1 - t) * p0 + t * p1
        base = base + perp * signed_uniform(gen, 0.3, 1.0) * sag_frac * length
        base[2] = base[2] + signed_uniform(gen, 0.0, axial_wobble_nm)
        ctrl.append(base)
    ctrl.append(p1)
    ctrl = torch.stack(ctrl)
    n_out = max(8, int(length / 20.0))  # ~1 point every 20 nm
    ts = torch.linspace(0, len(ctrl) - 1, n_out, dtype=torch.float64)
    lo = ts.floor().long().clamp(max=len(ctrl) - 2)
    frac = (ts - lo).unsqueeze(1)
    return (1 - frac) * ctrl[lo] + frac * ctrl[lo + 1]


class Filament3D(SyntheticObjectType):
    """Thin curved fibres (actin filaments / microtubules)."""

    name = "Filament"
    toml_key = "filament"

    ui_params = {
        "count": count_param("filaments", default=6),
        "radius_nm": value_param("Fibre radius", "r", 50.0, "nm"),
        "length_min_nm": value_param("Length min", "\\ell^{min}", 1000.0, "nm"),
        "length_max_nm": value_param("Length max", "\\ell^{max}", 4000.0, "nm"),
        "sag_frac": value_param("Lateral bending (frac)", "\\sigma_{bend}", 0.25),
        "axial_wobble_nm": value_param("Axial wobble", "\\Delta z_{wobble}", 40.0, "nm"),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
        "depth_min_frac": value_param("Depth min (frac)", "z^{min}", 0.0),
        "depth_max_frac": value_param("Depth max (frac)", "z^{max}", 0.7),
    }

    @classmethod
    def sample(cls, params, gen, grid) -> Filament:
        p = params
        span = grid.zN_nm - grid.z0_nm
        z_lo = grid.z0_nm + float(p["depth_min_frac"]) * span
        z_hi = grid.z0_nm + float(p["depth_max_frac"]) * span
        # random start point in the field
        x0 = uniform(gen, 0.0, grid.Lx_nm).item()
        y0 = uniform(gen, 0.0, grid.Ly_nm).item()
        z0 = uniform(gen, z_lo, z_hi).item()
        # random in-plane direction and length -> end point clamped into the field
        theta = uniform(gen, 0.0, 6.283185307).item()
        length = uniform(gen, float(p["length_min_nm"]), float(p["length_max_nm"])).item()
        x1 = min(max(x0 + length * torch.cos(torch.tensor(theta)).item(), 0.0), grid.Lx_nm)
        y1 = min(max(y0 + length * torch.sin(torch.tensor(theta)).item(), 0.0), grid.Ly_nm)
        z1 = uniform(gen, z_lo, z_hi).item()
        pts = _curve_points(gen, (x0, y0, z0), (x1, y1, z1),
                            sag_frac=float(p["sag_frac"]),
                            axial_wobble_nm=float(p["axial_wobble_nm"]))
        return Filament(points_nm=pts, radius_nm=float(p["radius_nm"]),
                        amplitude=float(p["amplitude"]))
