"""
Plug-and-Play (PnP) algorithm for 3D MA-TIRF reconstruction.

HQS inversion step:
    f = (H^T H + alpha I)^{-1} (H^T g + alpha z)

H^T H is small (nz x nz), so we diagonalise it once (symmetric eigendecomposition
H^T H = Q diag(w) Q^T) and reuse it every iteration: for any alpha the solve is
    (H^T H + alpha I)^{-1} y = Q diag(1 / (w + alpha)) Q^T y
which avoids re-inverting the matrix at each of the (varying-alpha) iterations.
"""

import copy

import torch

from common.algorithms import BasePnp
from matirf.core.operations import apply_matirf_operator
from matirf.gui.estimate_delta import estimate_delta
import common.settings as settings


class PnpAlgo(BasePnp):
    """PnP for 3D MA-TIRF. HQS inversion via a cached eigendecomposition of H^T H."""

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

    def precompute(self, g, H, params):
        """Diagonalise H^T H once and cache H^T g for the (varying-alpha) HQS solves."""
        HtH = H.transpose(0, 1) @ H
        self._eigvals, self._eigvecs = torch.linalg.eigh(HtH)  # HtH = Q diag(w) Q^T
        self._Htg = self.apply_adjoint(H, g)

    def solve_hqs(self, H, g, z, alpha, params):
        """f = Q diag(1 / (w + alpha)) Q^T (H^T g + alpha z)"""
        y = self._Htg + alpha * z                      # (nz, ny, nx)
        y_flat = y.reshape(y.shape[0], -1)             # (nz, ny*nx)
        coeff = self._eigvecs.transpose(0, 1) @ y_flat  # Q^T y
        coeff = coeff / (self._eigvals + alpha).unsqueeze(1)
        out = self._eigvecs @ coeff                    # Q (...)
        return out.reshape(y.shape)
