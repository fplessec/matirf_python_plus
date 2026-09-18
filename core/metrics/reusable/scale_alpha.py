"""
Scale alignment factor alpha* = argmin ||f_true - alpha * f||^2.

Only meaningful for scale-ambiguous inverse problems where the
reconstruction is defined up to a multiplicative constant.
"""

from core.metrics.base import Metric, optimal_scale
from core.features import Feature


class ScaleAlpha(Metric):

    name = "Scale_alpha"
    requires = {Feature.SCALE_AMBIGUOUS}

    def compute(self, f, f_true, features=set(), **kw):
        return optimal_scale(f, f_true)
