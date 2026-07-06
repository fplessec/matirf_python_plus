"""
Plug-and-Play (PnP) algorithm for 3D MA-TIRF reconstruction.

HQS inversion step solved by matrix inversion:
    f = (H^T H + alpha I)^{-1} (H^T g + alpha z)
"""

import copy

import torch

from common.algorithms import BasePnp
from matirf.core.operations import apply_matirf_operator
from matirf.gui.estimate_delta import estimate_delta
import common.settings as settings


class PnpAlgo(BasePnp):
    """PnP for 3D MA-TIRF. HQS inversion via matrix inverse."""

    supported_features = {"3d", "anisotropic"}

    ui_params = copy.deepcopy(BasePnp.ui_params)
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

    def solve_hqs(self, H, g, z, alpha, params):
        """f = (H^T H + alpha I)^{-1} (H^T g + alpha z)"""
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device)
        return apply_matirf_operator(
            torch.inverse(HtH + alpha * identity),
            Htg + alpha * z
        )
