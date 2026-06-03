from typing import Dict, Any
import time

import torch
import torch.nn.functional as F

from base import Algorithm
from algorithms.denoisers import denoise_tv_bregman, denoise_NL_RIDGE
from deconv.core.operations import apply_psf, get_variables_from_dict
from settings import device, dtype


class McmcAlgo(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):
        self.fixe_randomness()

        (max_iter, step_size, beta, sigma, K, denoiser) = get_variables_from_dict(
            params, ['max_iter', 'step_size', 'beta', 'sigma', 'K', 'denoiser'])

        t0 = time.time()

        # f0 = g (for deconv, f and g have the same shape)
        f = g.clone()
        accepted = 0

        self._print(f"MCMC start | T={max_iter} | step_size={step_size} | sigma={sigma} | beta={beta}")
        self._print(f"denoiser={denoiser}")

        for k in range(max_iter):
            if self.is_stop_requested():
                return None

            z = self.proposal_step(f, g, H, step_size, sigma, denoiser)
            is_accepted = self.evaluation_step(f, z, g, H, beta)

            if is_accepted:
                f = z
                accepted += 1

            if k % K == K - 1:
                rate = accepted / (k + 1) * 100
                loss_f = 0.5 * F.mse_loss(apply_psf(H, f), g).item()
                self._print(
                    f"iter {k + 1:4d} | accepted={accepted}/{k + 1} ({rate:.0f}%) | "
                    f"loss={loss_f:.3e}"
                )

        rate = accepted / max_iter * 100
        self._print(f"\naccepted {accepted}/{max_iter} ({rate:.1f}%)")
        self._print(f"execution in {time.time() - t0:.2f} s")
        return f

    ## proposes z from f via gradient step + noise + positivity + denoising:
    def proposal_step(self, f, g, H, step_size, sigma, denoiser):
        # 1. gradient step on the data fidelity 1/2 ||Hf - g||^2:
        #    grad = H^T(Hf - g)
        residual = apply_psf(H, f) - g
        grad = apply_psf(H, residual, adjoint=True)
        z = f - step_size * grad

        # 2. stochastic perturbation:
        z = z + sigma * torch.randn_like(z)

        # 3. positivity constraint:
        z = z.clamp(min=0.)

        # 4. denoising (implicit prior):
        z = self.apply_denoiser(z, sigma, denoiser)

        return z

    ## accepts or rejects z with Metropolis-Hastings ratio on the data fidelity:
    def evaluation_step(self, f, z, g, H, beta):
        # D(x) = 1/2 ||Hx - g||^2
        Hf = apply_psf(H, f)
        Hz = apply_psf(H, z)
        D_f = 0.5 * F.mse_loss(Hf, g, reduction='sum')
        D_z = 0.5 * F.mse_loss(Hz, g, reduction='sum')

        # acceptance probability: a = min(1, exp((D_f - D_z) / beta))
        log_ratio = (D_f - D_z) / beta
        acceptance_prob = torch.min(torch.tensor(1.0), torch.exp(log_ratio))

        alpha = torch.rand(1)
        return alpha.item() <= acceptance_prob.item()

    ## applies the selected denoiser to z:
    def apply_denoiser(self, z, sigma, denoiser):
        original_shape = z.shape
        if denoiser == "None":
            return z
        elif denoiser == "Non-Local Ridge":
            result = denoise_NL_RIDGE(z, sigma)
        elif denoiser == "TV Bregman":
            # weight = 1/sigma: stronger denoising for larger noise
            weight = max(1.0 / sigma, 0.1)
            result = denoise_tv_bregman(z, weight=weight)
        else:
            self._print(f"denoiser '{denoiser}' not implemented, skipping")
            return z
        # denoisers may add/change dimensions, reshape back to original
        return result.reshape(original_shape)
