"""
Plug-and-Play ADMM (PnP-ADMM) for 3D MA-TIRF reconstruction.

The x-step linear system (H^T H + rho I)^{-1} (...) is solved with a cached
eigendecomposition of the small (nz x nz) matrix H^T H, so rho can vary for free
and the matrix is never re-inverted inside the loop.
"""

import torch

from common.algorithms import BasePnpAdmm
from matirf.algorithms._base import MatirfForwardModel, with_delta_estimate_button


class PnpAdmmAlgo(MatirfForwardModel, BasePnpAdmm):
    """PnP-ADMM for 3D MA-TIRF. Linear solve via a cached eigendecomposition of H^T H."""

    ui_params = with_delta_estimate_button(BasePnpAdmm.ui_params)

    def precompute(self, g, H, params):
        """Diagonalise H^T H once: H^T H = Q diag(w) Q^T (symmetric)."""
        HtH = H.transpose(0, 1) @ H
        self._eigvals, self._eigvecs = torch.linalg.eigh(HtH)

    def solve_linear_system(self, Htg, v, rho, params):
        """(H^T H + rho I)^{-1} (Htg + rho v) = Q diag(1 / (w + rho)) Q^T (Htg + rho v)."""
        y = Htg + rho * v                              # (nz, ny, nx)
        y_flat = y.reshape(y.shape[0], -1)             # (nz, ny*nx)
        coeff = self._eigvecs.transpose(0, 1) @ y_flat  # Q^T y
        coeff = coeff / (self._eigvals + rho).unsqueeze(1)
        out = self._eigvecs @ coeff                    # Q (...)
        return out.reshape(y.shape)
