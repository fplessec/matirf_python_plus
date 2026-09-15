"""
PPXA algorithm for 3D MA-TIRF reconstruction.

Data fidelity prox solved by matrix inversion:
    p = (I + gamma*(1-lambda_reg)*H^T H)^{-1} (u + gamma*(1-lambda_reg)*H^T g)

Regularization prox delegated to the Regularization object from common.
"""

import torch

from common.algorithms import BasePpxa
from common.utils import get_variables_from_dict
from matirf.core.operations import apply_matirf_operator
from matirf.algorithms._base import MatirfForwardModel, with_delta_estimate_button
import common.settings as settings


class PpxaAlgo(MatirfForwardModel, BasePpxa):
    """PPXA for 3D MA-TIRF. Data fidelity prox via matrix inverse."""

    ui_params = with_delta_estimate_button(BasePpxa.ui_params)

    def precompute(self, g, H, params):
        """Precompute matrices for the data fidelity prox."""
        (gamma, lambda_reg) = get_variables_from_dict(params, ['gamma', 'lambda_reg'])

        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        identity = torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device)

        gamma_prox_estimated = 1 / torch.linalg.norm(H)**2 / (1 - lambda_reg)
        self._print(f"gamma prox estimation = {gamma_prox_estimated}\n")

        self._P = torch.inverse(identity + gamma * (1 - lambda_reg) * HtH)
        self._Htg = apply_matirf_operator(Ht, g)
        self._gamma = gamma
        self._lambda_reg = lambda_reg

    def init_f(self, g, H, params):
        """f0 = (H^T H + lambda_rr I)^{-1} H^T g   (ridge regression)."""
        return self.ridge_inverse(g, H)

    def compute_data_prox(self, u, params):
        """p = (I + gamma*(1-lambda_reg)*H^T H)^{-1} (u + gamma*(1-lambda_reg)*H^T g)"""
        return apply_matirf_operator(
            self._P,
            u + self._gamma * (1 - self._lambda_reg) * self._Htg
        )
