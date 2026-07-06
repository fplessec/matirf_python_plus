"""
PPXA algorithm for 3D MA-TIRF reconstruction.

Data fidelity prox solved by matrix inversion:
    p = (I + gamma*(1-lambda_reg)*H^T H)^{-1} (u + gamma*(1-lambda_reg)*H^T g)

Regularization prox delegated to the Regularization object from common.
"""

import copy

import torch

from common.algorithms import BasePpxa
from common.utils import get_variables_from_dict
from matirf.core.operations import apply_matirf_operator
from matirf.gui.estimate_delta import estimate_delta
import common.settings as settings


class PpxaAlgo(BasePpxa):
    """PPXA for 3D MA-TIRF. Data fidelity prox via matrix inverse."""

    supported_features = {"3d", "anisotropic"}

    # patch ui_params: add estimate_delta button on delta parameter
    ui_params = copy.deepcopy(BasePpxa.ui_params)
    ui_params["delta"]["extra_button"] = {
        "label": "Estimate",
        "tooltip": "Estimate delta from measurement parameters (.json) "
                   "and operator parameters (nz, z0, zN).",
        "callback": estimate_delta,
    }

    def apply_forward(self, H, f):
        return apply_matirf_operator(H, f)

    def apply_adjoint(self, H, x):
        return apply_matirf_operator(H.transpose(0, 1), x)

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
        """f0 = ( HtH + lambda_rr Id )^(-1) Htg   (ridge regression)"""
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device)
        # lambda_rr >> 1 to stabilize inversion
        lambda_rr = 10000.
        return apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Htg)

    def compute_data_prox(self, u, params):
        """p = (I + gamma*(1-lambda_reg)*H^T H)^{-1} (u + gamma*(1-lambda_reg)*H^T g)"""
        return apply_matirf_operator(
            self._P,
            u + self._gamma * (1 - self._lambda_reg) * self._Htg
        )
