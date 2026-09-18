"""
Generic Plug-and-Play (PnP) algorithm for inverse problems.

This is Plug-and-Play *Half-Quadratic Splitting* (HQS): it alternates a data
inversion step and a denoising step, with no dual variable. For the augmented
version with a Lagrange multiplier, see BasePnpAdmm (Plug-and-Play ADMM).

Alternates between:
    1. Inversion step (Half-Quadratic Splitting):
       f = argmin_f  1/2 ||Hf - g||^2 + alpha/2 ||f - z||^2
       i.e. f = (H^t H + alpha I)^{-1} (H^t g + alpha z)
    2. Denoising step (implicit prior):
       z = D_sigma(f)

Supports Kai Zhang (DPIR) scheduling: alpha increases and sigma decreases
logarithmically across iterations to improve convergence.

Scale bridge:
    the reconstruction f lives in [0, 1], but the denoisers are tuned for the
    classic [0, 255] intensity convention (e.g. NL-Ridge branches on sigma<=15,
    TV Bregman uses weight=1/sigma). We therefore denoise on the native 255
    scale ``D(f * 255, sigma) / 255`` and express sigma on that same scale, as
    in Zhang et al. (DPIR). For purely spatial denoisers (Gaussian, Bilateral)
    this scaling is a harmless no-op.

Subclasses must override:
    apply_forward(H, f)               : computes Hf
    apply_adjoint(H, x)               : computes H^t x
    solve_hqs(H, g, z, alpha, params) : solves the HQS inversion step
Subclasses may override:
    precompute(g, H, params)          : cache operators reused every iteration
"""

from typing import Dict, Any
import time

import numpy as np
import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.pnp.ui_params import PNP_UI_PARAMETERS
from common.denoisers import DENOISER_REGISTRY
from common.utils import get_variables_from_dict


## denoisers are calibrated for the [0, 255] intensity convention:
_DENOISER_SCALE = 255.0


class BasePnp(Algorithm):

    name = "PNP"
    ui_params = PNP_UI_PARAMETERS
    estimator_type = "MAP"
    uses_denoiser = True
    uses_regularization = False

    def run(self, g, H, params: Dict[str, Any]):
        self.fix_randomness()

        (sigma, n_iter, denoiser, kai_zhang, lambda_kz, forced_pos) = get_variables_from_dict(
            params, ['sigma', 'iter', 'denoiser', 'kai_zhang', 'lambda_kz', 'forced_pos'])
        ## delta is only present for anisotropic problems (matirf); default isotropic:
        delta = params.get('delta', 1.0)

        self.precompute(g, H, params)

        z = self.init_z(g, H, params)

        self._warn_if_slice_by_slice(denoiser, z)

        # alpha / sigma schedules (sigma is on the [0, 255] denoiser scale):
        if kai_zhang:
            sigma_1, sigma_K = sigma, 1
            alpha_1 = lambda_kz * sigma_K ** 2 / sigma_1 ** 2
            alpha_K = lambda_kz
            alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_K), n_iter)
            sigmas = np.sqrt(lambda_kz / alphas) * sigma_K
        else:
            alphas = [lambda_kz] * n_iter
            sigmas = [sigma] * n_iter

        denoiser_fn = None if denoiser == "None" else DENOISER_REGISTRY.get(denoiser)
        if denoiser != "None" and denoiser_fn is None:
            self._print(f"denoiser '{denoiser}' not found in registry, stopping")
            return None

        t0 = time.time()

        f = z
        for k in range(n_iter):
            if self.is_stop_requested():
                return None

            self._print(
                f"[PnP] iter {k + 1}/{n_iter} | "
                f"alpha = {alphas[k]:.3g} | sigma = {sigmas[k]:.3g}"
            )

            # 1. inversion step (HQS):
            f = self.solve_hqs(H, g, z, alphas[k], params)

            # 2. denoising step (implicit prior), on the native [0, 255] scale:
            if denoiser_fn is None:
                z = f.clone()
            else:
                z = self._denoise(denoiser_fn, f, sigmas[k], delta)

            if forced_pos:
                f = f.clamp(min=0)
                z = z.clamp(min=0)

            self._update_figure(f)

        self._print(f"execution in {time.time() - t0:.2f} s")
        return f

    @staticmethod
    def _denoise(denoiser_fn, f, sigma, delta):
        """Denoises f on the denoiser's native [0, 255] scale, back to f's scale."""
        return (denoiser_fn(f * _DENOISER_SCALE, sigma, delta) / _DENOISER_SCALE).reshape(f.shape)

    def precompute(self, g, H, params):
        """Precomputes operators reused by solve_hqs (e.g. an eigendecomposition)."""
        pass

    def init_z(self, g, H, params):
        """Initialization of z (denoised variable). Warm start with the back-projection H^t g."""
        return self.apply_adjoint(H, g).clamp(min=0)

    def solve_hqs(self, H, g, z, alpha, params):
        """
        Solves the HQS inversion step:
            f = (H^t H + alpha I)^{-1} (H^t g + alpha z)
        Must be overridden by subclasses (matrix inversion, Fourier, etc.).
        """
        raise NotImplementedError
