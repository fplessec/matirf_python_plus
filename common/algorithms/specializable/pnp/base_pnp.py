"""
Generic Plug-and-Play (PnP) algorithm for inverse problems.

Alternates between:
    1. Inversion step (Half-Quadratic Splitting):
       f = argmin_f  1/2 ||Hf - g||^2 + alpha/2 ||f - z||^2
       i.e. f = (H^T H + alpha I)^{-1} (H^T g + alpha z)
    2. Denoising step (implicit prior):
       z = D_sigma(f)

Supports Kai Zhang scheduling: alpha and sigma decrease/increase
logarithmically across iterations to improve convergence.

Subclasses must override:
    apply_forward(H, f)           : computes Hf
    apply_adjoint(H, x)          : computes H^T x
    solve_hqs(H, g, z, alpha, params) : solves the HQS inversion step
"""

from typing import Dict, Any
import time

import numpy as np
import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.pnp.ui_params import PNP_UI_PARAMETERS
from common.denoisers import DENOISER_REGISTRY
from common.utils import get_variables_from_dict


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

        z = self.init_z(g, H, params)
        self._warn_if_slice_by_slice(denoiser, z)

        # alpha / sigma schedules:
        if kai_zhang:
            sigma_1, sigma_K = sigma, 1.0
            alpha_1 = lambda_kz * sigma_K ** 2 / sigma_1 ** 2
            alpha_K = lambda_kz
            alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_K), n_iter)
            sigmas = np.sqrt(lambda_kz / alphas) * sigma_K
        else:
            alphas = [lambda_kz] * n_iter
            sigmas = [sigma] * n_iter

        t0 = time.time()

        for k in range(n_iter):
            if self.is_stop_requested():
                return None

            self._print(
                f"[PnP] iter {k + 1}/{n_iter} | "
                f"alpha = {alphas[k]:.3g} | sigma = {sigmas[k]:.3g}"
            )

            # 1. inversion step (HQS):
            f = self.solve_hqs(H, g, z, alphas[k], params)

            # 2. denoising step (implicit prior):
            if denoiser == "None":
                z = f.clone()
            else:
                denoiser_fn = DENOISER_REGISTRY.get(denoiser)
                if denoiser_fn is None:
                    self._print(f"denoiser '{denoiser}' not found in registry, stopping")
                    break
                z = denoiser_fn(f, sigmas[k]).reshape(f.shape)

            if forced_pos:
                f = f.clamp(min=0)
                z = z.clamp(min=0)

            self._update_figure(f)

        self._print(f"execution in {time.time() - t0:.2f} s")
        return f

    def init_z(self, g, H, params):
        """Initialization of z (denoised variable). Default: zeros like H^T g."""
        Htg = self.apply_adjoint(H, g)
        return torch.zeros_like(Htg)

    def solve_hqs(self, H, g, z, alpha, params):
        """
        Solves the HQS inversion step:
            f = (H^T H + alpha I)^{-1} (H^T g + alpha z)
        Must be overridden by subclasses (matrix inversion, Fourier, etc.).
        """
        raise NotImplementedError
