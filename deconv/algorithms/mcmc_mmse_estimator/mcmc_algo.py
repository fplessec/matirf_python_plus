"""
MCMC MMSE estimator for 2D deconvolution.

Two proposal methods available (selectable via UI):
    C: noise → data consistency → positivity → denoising
    D: data consistency → noise → positivity → denoising

Data consistency uses Wiener/ridge inversion in Fourier domain:
    H_inv(x) = F^{-1}[ conj(H_fft) / (|H_fft|^2 + lambda_rr) * F(x) ]
"""

import copy

import torch

from common.algorithms import BaseMcmc
from deconv.core.operations import apply_psf, _psf_to_fft_kernel
from deconv.gui.estimate_lambda_rr import estimate_lambda_rr


PROPOSAL_METHODS = ["D", "C"]


class McmcAlgo(BaseMcmc):
    """MCMC for 2D deconvolution. Forward operator is FFT-based convolution."""

    supported_features = {"2d"}

    ui_params = copy.deepcopy(BaseMcmc.ui_params)
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
            "default": 0.01,
        },
        "extra_button": {
            "label": "Estimate",
            "tooltip": "Estimate lambda_rr from the Fourier spectrum of the PSF.\n"
                       "Displays the sorted |H_fft| values and lets you\n"
                       "choose which frequency to use as cutoff.",
            "callback": estimate_lambda_rr,
        }
    }

    def apply_forward(self, H, f):
        return apply_psf(H, f)

    def apply_adjoint(self, H, x):
        return apply_psf(H, x, adjoint=True)

    def _wiener_inverse(self, H, x, lambda_rr):
        """Wiener/ridge deconvolution in Fourier domain:
        (H^T H + lambda_rr I)^{-1} H^T x."""
        H_fft = torch.fft.fft2(_psf_to_fft_kernel(H, x.shape))
        x_fft = torch.fft.fft2(x)
        wiener_filter = H_fft.conj() / (H_fft.abs() ** 2 + lambda_rr)
        return torch.real(torch.fft.ifft2(wiener_filter * x_fft))

    def proposal_step(self, f, g, H, sigma, denoiser, params):
        method = params.get('proposal_method', 'D')
        lambda_rr = params.get('lambda_rr', 0.01)

        if method == 'C':
            return self._proposal_C(f, g, H, sigma, denoiser, lambda_rr)
        else:
            return self._proposal_D(f, g, H, sigma, denoiser, lambda_rr)

    def _proposal_C(self, f, g, H, sigma, denoiser, lambda_rr):
        """Noise → data consistency → positivity → denoising."""
        z = f + sigma * torch.randn_like(f)
        residual = g - self.apply_forward(H, z)
        z = z + self._wiener_inverse(H, residual, lambda_rr)
        z.clamp_(min=0)
        z = self.apply_denoiser(z, sigma, denoiser)
        return z

    def _proposal_D(self, f, g, H, sigma, denoiser, lambda_rr):
        """Data consistency → noise → positivity → denoising."""
        residual = g - self.apply_forward(H, f)
        z = f + self._wiener_inverse(H, residual, lambda_rr)
        z = z + sigma * torch.randn_like(z)
        z.clamp_(min=0)
        z = self.apply_denoiser(z, sigma, denoiser)
        return z
