"""
MCMC MMSE estimator for 3D MA-TIRF reconstruction.

Two proposal methods available (selectable via UI):
    C: noise → data consistency → positivity → denoising
    D: data consistency → noise → positivity → denoising
"""

import torch

from common.algorithms import BaseMcmc
from matirf.algorithms._base import MatirfForwardModel, with_delta_estimate_button
from matirf.gui.estimate_lambda_rr import estimate_lambda_rr


PROPOSAL_METHODS = ["D", "C"]


class McmcAlgo(MatirfForwardModel, BaseMcmc):
    """MCMC for 3D MA-TIRF. Forward operator is matrix multiplication."""

    ui_params = with_delta_estimate_button(BaseMcmc.ui_params)
    del ui_params["step_size"]
    del ui_params["mu"]
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

    def init_f(self, g, H, params):
        """f0 = H^T g  (back-projection into reconstruction space)."""
        return self.apply_adjoint(H, g).detach()

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
        z = z + self.ridge_inverse(residual, H, lambda_rr)
        z.clamp_(min=0)
        z = self.apply_denoiser(z, sigma, denoiser)
        return z

    def _proposal_D(self, f, g, H, sigma, denoiser, lambda_rr):
        """Data consistency → noise → positivity → denoising."""
        residual = g - self.apply_forward(H, f)
        z = f + self.ridge_inverse(residual, H, lambda_rr)
        z = z + sigma * torch.randn_like(z)
        z.clamp_(min=0)
        z = self.apply_denoiser(z, sigma, denoiser)
        return z
