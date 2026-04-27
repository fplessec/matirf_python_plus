from typing import Dict, Any
import time

import torch

from ..abstract_algo import Algorithm
from ..denoisers import denoise_tv_bregman, denoise_NL_RIDGE
from core.operations import apply_matirf_operator, get_variables_from_dict, estimate_delta_anisotropy_from_params
from settings import device, dtype


class McmcAlgo(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):
        self.fixe_randomness()

        (max_iter, mu, beta, sigma, EPS, denoiser) = get_variables_from_dict(
            params, ['max_iter', 'mu', 'beta', 'sigma', 'EPS', 'denoiser'])

        #  f = ( HtH + gamma Id )^(-1) Htg
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        HH_inv = torch.inverse(HtH + mu * identity)
        f = apply_matirf_operator(HH_inv, Htg)

        t0 = time.time()
        for k in range(max_iter):
            if self.is_stop_requested():
                return None
            self._print(f"mcmc_iter = {k + 1}")
            # Inversion:
            f = apply_matirf_operator(HH_inv, Htg + mu * f)
            # Projection sur f >= 0
            with torch.no_grad():
                f.clamp_(min=0)
            # Perturbation gaussian nosie std sigma:
            f += sigma * torch.randn_like(f)
            # Denoizing:
            if denoiser == "None":
                pass
            elif denoiser == "Non-Local Ridge":
                f = denoise_NL_RIDGE(f, sigma)
            elif denoiser == "TV Bregman":
                f = denoise_tv_bregman(f, weight=sigma)
            else:
                self._print(f"{denoiser} not implemented.")
                break
        with torch.no_grad():
            f.clamp_(min=0)
        self._print(f"execution in {time.time() - t0} s")
        return f