"""
Adam optimizer for 3D MA-TIRF reconstruction.

Minimizes:  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

Initialization: f0 = (H^T H + lambda_rr I)^{-1} H^T g  (ridge regression).
"""

import copy

import torch

from common.algorithms import BaseAdam
from common.utils import get_variables_from_dict
from matirf.core.operations import apply_matirf_operator
from matirf.gui.estimate_delta import estimate_delta
import common.settings as settings


class AdamAlgo(BaseAdam):
    """Adam for 3D MA-TIRF. Forward operator is matrix multiplication."""

    supported_features = {"3d", "anisotropic"}

    # patch ui_params: add estimate_delta button on delta parameter
    ui_params = copy.deepcopy(BaseAdam.ui_params)
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

    def init_f(self, g, H, params):
        """f0 = ( HtH + lambda_rr Id )^(-1) Htg   (ridge regression)"""
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device)
        # lambda_rr >> 1 to stabilize inversion
        lambda_rr = 10000.
        f = apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Htg)
        return f.detach()
