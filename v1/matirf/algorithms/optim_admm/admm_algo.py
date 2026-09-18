"""
ADMM algorithm for 3D MA-TIRF reconstruction.

https://arxiv.org/pdf/1801.00882
https://github.com/zcshinee/Pol-TIRF/tree/master
"""

import torch

from common.algorithms import BaseAdmm
from matirf.core.operations import apply_matirf_operator
from matirf.algorithms._base import MatirfForwardModel
import common.settings as settings


class AdmmAlgo(MatirfForwardModel, BaseAdmm):
    """ADMM for 3D MA-TIRF. Linear system solved by matrix inverse."""

    def precompute(self, g, H, params):
        """Precompute (HtH + mu I)^{-1} for the u-step."""
        mu = params['mu']
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        identity = torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device)
        self._HH_inv = torch.inverse(HtH + mu * identity)

    def init_f(self, g, H, params):
        """f0 = (HtH + mu I)^{-1} Htg"""
        Htg = self.apply_adjoint(H, g)
        return apply_matirf_operator(self._HH_inv, Htg)

    def solve_linear_system(self, Htg, v, mu, params):
        """(H^T H + mu I)^{-1} (Htg + mu v)"""
        return apply_matirf_operator(self._HH_inv, Htg + mu * v)

    def compute_threshold(self, g, H, params):
        return g.max() / H.max() * 0.1
