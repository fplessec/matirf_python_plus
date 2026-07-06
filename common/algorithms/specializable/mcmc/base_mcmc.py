"""
Generic MCMC MMSE estimator for inverse problems.

Generates samples from the posterior pi(f) proportional to exp(-U(f,g)/2)
using Metropolis-Hastings, and returns the MMSE estimate:
    f_MMSE ≈ 1/T * sum_{t=1}^{T} f_t   (over accepted samples)

The integral is intractable, but by the law of large numbers the
empirical mean of the accepted samples converges in probability to f_MMSE.

Subclasses must override:
    apply_forward(H, f)       : computes Hf
    apply_adjoint(H, x)       : computes H^T x
    proposal_step(f, g, H, sigma, denoiser, params) : generates candidate z
"""

from typing import Dict, Any
import time

import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.mcmc.ui_params import MCMC_UI_PARAMETERS
from common.denoisers import DENOISER_REGISTRY
from common.utils import get_variables_from_dict


class BaseMcmc(Algorithm):

    name = "MCMC"
    ui_params = MCMC_UI_PARAMETERS
    estimator_type = "MMSE"
    uses_denoiser = True
    uses_regularization = False

    def run(self, g, H, params: Dict[str, Any]):
        self.fix_randomness()

        (max_iter, beta, sigma, K, denoiser) = get_variables_from_dict(
            params, ['max_iter', 'beta', 'sigma', 'K', 'denoiser'])

        self._data_fidelity = self._create_data_fidelity(params)

        t0 = time.time()

        f = self.init_f(g, H, params)
        f_sum = torch.zeros_like(f)
        accepted = 0

        self._print(f"MCMC start | T={max_iter} | sigma={sigma} | beta={beta}")
        self._print(f"denoiser={denoiser}")
        self._warn_if_slice_by_slice(denoiser, f)

        for k in range(max_iter):
            if self.is_stop_requested():
                return None

            z = self.proposal_step(f, g, H, sigma, denoiser, params)
            is_accepted = self.evaluation_step(f, z, g, H, beta)

            if is_accepted:
                f = z
                accepted += 1
                f_sum += f

            if k % K == K - 1:
                rate = accepted / (k + 1) * 100
                Hf = self.apply_forward(H, f)
                loss_f = self._data_fidelity.loss(Hf, g).item()
                self._print(
                    f"iter {k + 1:4d} | accepted={accepted}/{k + 1} ({rate:.0f}%) | "
                    f"loss={loss_f:.3e}"
                )
                ## for MCMC, the live preview shows the running MMSE estimate:
                if accepted > 0:
                    self._update_figure(f_sum / accepted)
                else:
                    self._update_figure(f)

        rate = accepted / max_iter * 100
        self._print(f"\naccepted {accepted}/{max_iter} ({rate:.1f}%)")
        self._print(f"execution in {time.time() - t0:.2f} s")

        # f_MMSE ≈ 1/T * sum(f_t accepted)
        if accepted > 0:
            return f_sum / accepted
        return f

    def init_f(self, g, H, params):
        """Initialization of f. Default: f0 = g."""
        return g.clone()

    ## accepts or rejects z with Metropolis-Hastings ratio on the data fidelity:
    def evaluation_step(self, f, z, g, H, beta):
        Hf = self.apply_forward(H, f)
        Hz = self.apply_forward(H, z)
        D_f = self._data_fidelity.loss(Hf, g)
        D_z = self._data_fidelity.loss(Hz, g)

        # acceptance probability: a = min(1, exp((D_f - D_z) / beta))
        log_ratio = (D_f - D_z) / beta
        acceptance_prob = torch.min(torch.tensor(1.0), torch.exp(log_ratio))

        alpha = torch.rand(1)
        return alpha.item() <= acceptance_prob.item()

    ## applies the selected denoiser to z:
    def apply_denoiser(self, z, sigma, denoiser):
        if denoiser == "None":
            return z
        denoiser_fn = DENOISER_REGISTRY.get(denoiser)
        if denoiser_fn is None:
            self._print(f"denoiser '{denoiser}' not found in registry, skipping")
            return z
        original_shape = z.shape
        result = denoiser_fn(z, sigma)
        return result.reshape(original_shape)

    def proposal_step(self, f, g, H, sigma, denoiser, params):
        """Generates a candidate z from f. Must be overridden by subclasses."""
        raise NotImplementedError
