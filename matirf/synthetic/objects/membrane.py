"""
Membrane — a thin, laterally corrugated sheet (membrane / adhesion patch).

The surface is a height field z_s(x, y) = z_center + Σ a_k·sin(2π(fx_k·x + fy_k·y) + φ_k)
with SIGNED spatial frequencies (so the corrugation is not biased toward one diagonal),
and the density falls off as a Gaussian of half-thickness 'thickness_nm' away from it,
modulated by a slow lateral envelope (patchiness). A small thickness makes it a strong
probe of axial super-resolution. The sheet's normal is ~along z, which is the physically
correct pose for a membrane inside a shallow slab (it cannot fold vertically).
"""

import math
from dataclasses import dataclass, field

import torch

from ..rng import uniform, signed_uniform
from .base import ContinuousObject, SyntheticObjectType, value_param, count_param


@dataclass
class MembraneSheet(ContinuousObject):
    z_center_nm: float
    thickness_nm: float
    amplitude: float = 1.0
    corrugation_amp_nm: float = 60.0
    n_waves: int = 3
    sharpness: float = 1.0
    _waves: list = field(default_factory=list)
    _env: list = field(default_factory=list)

    @classmethod
    def sample_surface(cls, gen, grid, z_center_nm, thickness_nm, amplitude=1.0,
                       corrugation_amp_nm=60.0, n_waves=3, sharpness=1.0):
        obj = cls(z_center_nm=z_center_nm, thickness_nm=thickness_nm, amplitude=amplitude,
                  corrugation_amp_nm=corrugation_amp_nm, n_waves=n_waves, sharpness=sharpness)
        L = max(grid.Lx_nm, grid.Ly_nm)
        for _ in range(n_waves):
            fx = signed_uniform(gen, 0.5, 2.0) / L   # signed -> no directional bias
            fy = signed_uniform(gen, 0.5, 2.0) / L
            phase = uniform(gen, 0.0, 2 * math.pi).item()
            obj._waves.append((fx, fy, phase, corrugation_amp_nm / max(1, n_waves)))
        for _ in range(2):
            obj._env.append((signed_uniform(gen, 0.3, 1.2) / L,
                             signed_uniform(gen, 0.3, 1.2) / L,
                             uniform(gen, 0.0, 2 * math.pi).item()))
        return obj

    def density(self, X, Y, Z):
        zs = torch.full_like(X, float(self.z_center_nm))
        for fx, fy, phase, a in self._waves:
            zs = zs + a * torch.sin(2 * math.pi * (fx * X + fy * Y) + phase)
        env = torch.ones_like(X)
        for fx, fy, phase in self._env:
            env = env * (0.6 + 0.4 * torch.sin(2 * math.pi * (fx * X + fy * Y) + phase))
        d2 = ((Z - zs) / self.thickness_nm) ** 2
        # sharpness = 1 -> Gaussian across the sheet; > 1 -> flat-topped, sharper edges
        if self.sharpness != 1.0:
            d2 = d2 ** self.sharpness
        return self.amplitude * env * torch.exp(-0.5 * d2)


class Membrane(SyntheticObjectType):
    """Thin corrugated sheets near the coverslip."""

    name = "Membrane"
    toml_key = "membrane"

    ui_params = {
        "count": count_param("membranes", default=0),
        "thickness_nm": value_param("Axial half-thickness", "\\tau", 30.0, "nm"),
        "corrugation_amp_nm": value_param("Corrugation amplitude", "a", 60.0, "nm"),
        "sharpness": value_param("Sharpness (1=Gaussian)", "s", 1.0),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
        "depth_min_frac": value_param("Depth min (frac)", "z^{min}", 0.1),
        "depth_max_frac": value_param("Depth max (frac)", "z^{max}", 0.7),
    }

    @classmethod
    def sample(cls, params, gen, grid) -> MembraneSheet:
        p = params
        span = grid.zN_nm - grid.z0_nm
        z_center = uniform(gen, grid.z0_nm + float(p["depth_min_frac"]) * span,
                           grid.z0_nm + float(p["depth_max_frac"]) * span).item()
        return MembraneSheet.sample_surface(
            gen, grid, z_center_nm=z_center, thickness_nm=float(p["thickness_nm"]),
            amplitude=float(p["amplitude"]), corrugation_amp_nm=float(p["corrugation_amp_nm"]),
            sharpness=float(p["sharpness"]))
