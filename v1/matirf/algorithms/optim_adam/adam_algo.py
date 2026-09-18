"""
Adam optimizer for 3D MA-TIRF reconstruction.

Minimizes:  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

Initialization: f0 = (H^T H + lambda_rr I)^{-1} H^T g  (ridge regression).
"""

from common.algorithms import BaseAdam
from matirf.algorithms._base import MatirfForwardModel, with_delta_estimate_button


class AdamAlgo(MatirfForwardModel, BaseAdam):
    """Adam for 3D MA-TIRF. Forward operator is matrix multiplication."""

    ui_params = with_delta_estimate_button(BaseAdam.ui_params)

    def init_f(self, g, H, params):
        """f0 = (H^T H + lambda_rr I)^{-1} H^T g   (ridge regression)."""
        return self.ridge_inverse(g, H).detach()
