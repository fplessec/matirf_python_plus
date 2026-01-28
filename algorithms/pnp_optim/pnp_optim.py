from typing import Dict, Any
import time
import numpy as np
import torch

from .nlbayes import denoise_NL_BAYES
from .nlridge import denoise_NL_RIDGE
from .cimg_denoise import denoise_Cimg
from ..abstract_algo import Algorithm
from settings import device, dtype
from operations import apply_matirf_operator, get_variables_from_dict

from skimage.restoration import estimate_sigma


class PnpOptimization(Algorithm):

    def run(self, g: torch.Tensor, H: torch.Tensor, params: Dict[str, Any]):

        (
            sigma,
            n_iter,
            denoiser,
            kai_zhang,
            lambda_kz,
            forced_pos,
            var_stab
        ) = get_variables_from_dict(
            params,
            ['sigma', 'iter', 'denoiser', 'kai_zhang', 'lambda_kz', 'forced_pos', 'var_stab']
        )

        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)

        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        z = torch.zeros_like(Htg, dtype=dtype, device=device)

        if kai_zhang:
            sigma_1, sigma_K = sigma, 1.0
            alpha_1 = lambda_kz * sigma_K**2 / sigma_1**2
            alpha_K = lambda_kz
            alpha = np.logspace(np.log10(alpha_1), np.log10(alpha_K), n_iter)
            sigma = np.sqrt(lambda_kz / alpha) * sigma_K
        else:
            alpha = [lambda_kz] * n_iter
            sigma = [sigma] * n_iter

        t0 = time.time()

        for k in range(n_iter):
            if self.is_stop_requested():
                return None

            self._print(
                f"[PnP] iter {k+1:}/{n_iter} | "
                f"alpha = {alpha[k]:.3g} | sigma = {sigma[k]:.3g}"
            )

            f = apply_matirf_operator(
                torch.inverse(HtH + alpha[k] * identity),
                Htg + alpha[k] * z
            )

            self._print(
                f"f : min = {f.min().item():.3g} | "
                f"max = {f.max().item():.3g} | "
                f"mean = {f.mean().item():.3g}"
            )

            _ = estimate_sigma(f, average_sigmas=True, channel_axis=0)

            if denoiser == "None":
                z = f.clone()
            elif denoiser == "Non-Local Ridge":
                z = denoise_NL_RIDGE(f, sigma[k])
            elif denoiser == "Non-Local Bayes":
                z = denoise_NL_BAYES(f, sigma[k])
            else:
                z = denoise_Cimg(f, sigma[k], denoiser)

            self._print(
                f"z : min = {z.min().item():.3g} | "
                f"max = {z.max().item():.3g} | "
                f"mean = {z.mean().item():.3g}"
            )

            if forced_pos:
                f = f.clamp(min=0)
                z = z.clamp(min=0)

        self._print(f"execution in {time.time() - t0} s")
        return f
