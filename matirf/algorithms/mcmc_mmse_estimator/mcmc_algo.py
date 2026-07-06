"""
MCMC MMSE estimator for 3D MA-TIRF reconstruction.

Two proposal methods available (selectable via UI):
    C: noise → data consistency → positivity → denoising
    D: data consistency → noise → positivity → denoising
"""

import copy

import torch

from common.algorithms import BaseMcmc
import common.settings as settings
from matirf.core.operations import apply_matirf_operator
from matirf.gui.estimate_delta import estimate_delta
from matirf.gui.estimate_lambda_rr import estimate_lambda_rr


PROPOSAL_METHODS = ["D", "C"]


class McmcAlgo(BaseMcmc):
    """MCMC for 3D MA-TIRF. Forward operator is matrix multiplication."""

    supported_features = {"3d", "anisotropic"}

    ui_params = copy.deepcopy(BaseMcmc.ui_params)
    del ui_params["step_size"]
    del ui_params["mu"]
    ui_params["delta"]["extra_button"] = {
        "label": "Estimate",
        "tooltip": "Estimate delta from measurement parameters (.json) "
                   "and operator parameters (nz, z0, zN).",
        "callback": estimate_delta,
    }
    ui_params["proposal_method"] = {
        "title": "Proposal method",
        "type": "option",
        "param_info": {
            "options_list": PROPOSAL_METHODS,
        }
    }
    ui_params["lambda_rr"] = {
        "title": "Ridge regularization (inversion)",
        "type": "value",
        "param_info": {
            "dtype": float,
            "unit": "",
            "latex_name": "\\lambda_{rr}",
            "default": 10000.0,
        },
        "extra_button": {
            "label": "Estimate",
            "tooltip": "Estimate lambda_rr from the singular value spectrum of H.\n"
                       "Computes the SVD of the MA-TIRF operator and lets you\n"
                       "choose which singular value to use as cutoff.",
            "callback": estimate_lambda_rr,
        }
    }

    def apply_forward(self, H, f):
        return apply_matirf_operator(H, f)

    def apply_adjoint(self, H, x):
        return apply_matirf_operator(H.transpose(0, 1), x)

    def init_f(self, g, H, params):
        """f0 = H^T g  (back-projection into reconstruction space)."""
        return self.apply_adjoint(H, g).detach()

    def _ridge_inverse(self, H, x, lambda_rr):
        """Computes (H^T H + lambda_rr I)^{-1} H^T x  (ridge pseudo-inverse).
        x must be in measurement space (n_angles, ...)."""
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htx = self.apply_adjoint(H, x)
        inv = torch.inverse(HtH + lambda_rr * torch.eye(H.shape[1], dtype=settings.dtype, device=settings.device))
        return apply_matirf_operator(inv, Htx)

    def proposal_step(self, f, g, H, sigma, denoiser, params):
        method = params.get('proposal_method', 'D')
        lambda_rr = params.get('lambda_rr', 10000.0)

        if method == 'C':
            return self._proposal_C(f, g, H, sigma, denoiser, lambda_rr)
        else:
            return self._proposal_D(f, g, H, sigma, denoiser, lambda_rr)

    def _proposal_C(self, f, g, H, sigma, denoiser, lambda_rr):
        """Noise → data consistency → positivity → denoising."""
        z = f + sigma * torch.randn_like(f)
        residual = g - self.apply_forward(H, z)
        z = z + self._ridge_inverse(H, residual, lambda_rr)
        z.clamp_(min=0)
        z = self.apply_denoiser(z, sigma, denoiser)
        return z

    def _proposal_D(self, f, g, H, sigma, denoiser, lambda_rr):
        """Data consistency → noise → positivity → denoising."""
        residual = g - self.apply_forward(H, f)
        z = f + self._ridge_inverse(H, residual, lambda_rr)
        z = z + sigma * torch.randn_like(z)
        z.clamp_(min=0)
        z = self.apply_denoiser(z, sigma, denoiser)
        return z
