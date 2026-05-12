from typing import Dict, Any
import time

import torch
import torch.nn.functional as F

from ..abstract_algo import Algorithm
from ..denoisers import denoise_tv_bregman, denoise_NL_RIDGE
from core.operations import apply_matirf_operator, get_variables_from_dict, estimate_delta_anisotropy_from_params
from settings import device, dtype


class McmcAlgo(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):
        self.fixe_randomness()

        (max_iter, mu, beta, sigma, EPS, denoiser) = get_variables_from_dict(
            params, ['max_iter', 'mu', 'beta', 'sigma', 'EPS', 'denoiser'])

        # gamma ridge regression:
        gamma = 1.
        gamma = mu

        #  f = ( HtH + gamma Id )^(-1) Htg
        Ht = H.transpose(0, 1)
        print(f"H.shape {H.shape}")
        print(f"Ht.shape {Ht.shape}")
        HtH = Ht @ H
        print(f"HtH.shape {HtH.shape}")
        Htg = apply_matirf_operator(Ht, g)
        print(f"Htg.shape {Htg.shape}")
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        HH_inv = torch.inverse(HtH + gamma * identity)
        print(f"HH_inv.shape {HH_inv.shape}")
        def inversion_1(input):
            """for input=x, return H^(-1).x but with pseudo inversion"""
            return apply_matirf_operator(HH_inv, apply_matirf_operator(Ht, input))

        #  f = H_pseudo_inv_RR . g
        H_pseudo_inv_RR = torch.inverse(H.t() @ H + gamma * identity) @ H.t()
        def inversion_2(input):
            """for input=x, return H^(-1).x but with pseudo inversion"""
            return apply_matirf_operator(H_pseudo_inv_RR, input)

        def inversion_3(input):
            return apply_matirf_operator(Ht, input)


        def score_diff(f, z):
            Hf = apply_matirf_operator(H, f)
            Hz = apply_matirf_operator(H, z)
            distance_f_g = F.mse_loss(Hf, g, reduction='mean')
            distance_z_g = F.mse_loss(Hz, g, reduction='mean')
            return distance_f_g - distance_z_g


        t0 = time.time()
        valid_proposal = []
        f = inversion_3(g.clone())
        f = g.clone()
        print(f"f0.shape {f.shape}")


        for k in range(max_iter):
            if self.is_stop_requested():
                return None
            self._print(f"mcmc_iter = {k + 1}")
            z = self.proposal_step(f, inversion_3, sigma, denoiser)
            is_valid = self.evaluation_step(f, z ,beta, score_diff)
            self._print(f"proposal is valid: {is_valid}")
            if is_valid:
                valid_proposal.append(z)
                f = z

        self._print(f"execution in {time.time() - t0} s")
        return f

    def proposal_step(self, f, inversion_fct, sigma, denoiser):
        # Inversion:
        z = inversion_fct(f)
        # Perturbation gaussian nosie std sigma:
        z += sigma * torch.randn_like(f)
        # Projection sur z >= 0
        with torch.no_grad():
            z.clamp_(min=0)
        # Denoizing:
        if denoiser == "None":
            pass
        elif denoiser == "Non-Local Ridge":
            z = denoise_NL_RIDGE(z, sigma)
        elif denoiser == "TV Bregman":
            z = denoise_tv_bregman(z, weight=sigma)
        else:
            self._print(f"{denoiser} not implemented.")
            z = None
        return z

    def evaluation_step(self, f, z, beta, score_diff_fct):
        acceptance_prob = torch.min(torch.tensor(1.), torch.exp( score_diff_fct(f,z) / beta ))
        alpha = torch.rand(1)
        return alpha <= acceptance_prob