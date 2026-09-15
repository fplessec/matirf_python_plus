"""
Shared MA-TIRF specialization bits, factored out so each algorithm file only carries
what is genuinely specific to *that* algorithm.

    MatirfForwardModel          the forward model (Hf / H^t g) — identical for every
                                MA-TIRF algorithm, so it lives here once. Also exposes a
                                ridge pseudo-inverse used by several algorithms' init_f.
    with_delta_estimate_button  wires the "Estimate" button onto the 'delta' UI parameter
                                (the same patch every MA-TIRF algorithm applied by hand).
"""

import copy

import torch

import common.settings as settings
from matirf.core.operations import apply_matirf_operator
from matirf.gui.estimate_delta import estimate_delta


class MatirfForwardModel:
    """The MA-TIRF forward model: shared by every MA-TIRF algorithm (mixin, no state)."""

    supported_features = {"3d", "anisotropic"}

    def apply_forward(self, H, f):
        return apply_matirf_operator(H, f)

    def apply_adjoint(self, H, x):
        return apply_matirf_operator(H.transpose(0, 1), x)

    def ridge_inverse(self, y, H, lambda_rr=10000.):
        """
        Ridge pseudo-inverse of a measurement-space tensor 'y':
            (H^T H + lambda_rr I)^{-1} H^T y
        lambda_rr >> 1 stabilizes the inversion. Used as a warm-start init_f and by MCMC's
        data-consistency step.
        """
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Hty = apply_matirf_operator(Ht, y)
        identity = torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device)
        return apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Hty)


def with_delta_estimate_button(base_ui):
    """
    Return a deep copy of 'base_ui' with an "Estimate" button wired onto the 'delta'
    parameter (estimates delta from the measurement .json + operator params nz/z0/zN).
    """
    ui = copy.deepcopy(base_ui)
    ui["delta"]["extra_button"] = {
        "label": "Estimate",
        "tooltip": "Estimate delta from measurement parameters (.json) "
                   "and operator parameters (nz, z0, zN).",
        "callback": estimate_delta,
    }
    return ui
