"""
DoubleLayer — two closely spaced thin sheets, an intentionally ill-conditioned probe.

When 'separation_nm' is comparable to or below the axial resolving power, the two layers
are hard to separate along z. This is the content that most strongly discriminates
reconstruction algorithms: a good reconstruction resolves two axial peaks, a poor one
merges them into a single blur.
"""

from dataclasses import dataclass

from ..rng import uniform
from .base import ContinuousObject, SyntheticObjectType, value_param, count_param
from .membrane import MembraneSheet


@dataclass
class DoubleLayer(ContinuousObject):
    top: MembraneSheet
    bot: MembraneSheet

    def density(self, X, Y, Z):
        return self.top.density(X, Y, Z) + self.bot.density(X, Y, Z)


class DoubleLayerType(SyntheticObjectType):
    """Two closely spaced sheets (ill-conditioned axial content)."""

    name = "Double layer"
    toml_key = "double_layer"

    ui_params = {
        "count": count_param("double layers", default=1),
        "separation_nm": value_param("Axial separation", "\\Delta z_{sep}", 70.0, "nm"),
        "thickness_nm": value_param("Axial half-thickness", "\\tau", 25.0, "nm"),
        "corrugation_amp_nm": value_param("Corrugation amplitude", "a", 30.0, "nm"),
        "sharpness": value_param("Sharpness (1=Gaussian)", "s", 1.0),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
        "depth_min_frac": value_param("Depth min (frac)", "z^{min}", 0.3),
        "depth_max_frac": value_param("Depth max (frac)", "z^{max}", 0.7),
    }

    @classmethod
    def sample(cls, params, gen, grid) -> DoubleLayer:
        p = params
        span = grid.zN_nm - grid.z0_nm
        z_center = uniform(gen, grid.z0_nm + float(p["depth_min_frac"]) * span,
                           grid.z0_nm + float(p["depth_max_frac"]) * span).item()
        sep = float(p["separation_nm"])
        th = float(p["thickness_nm"])
        amp = float(p["amplitude"])
        corr = float(p["corrugation_amp_nm"])
        sh = float(p["sharpness"])
        top = MembraneSheet.sample_surface(gen, grid, z_center + sep / 2, th, amp, corr,
                                           n_waves=2, sharpness=sh)
        bot = MembraneSheet.sample_surface(gen, grid, z_center - sep / 2, th, amp, corr,
                                           n_waves=2, sharpness=sh)
        return DoubleLayer(top=top, bot=bot)
