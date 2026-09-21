"""
MCMC — an MMSE estimator by posterior sampling.

Every other solver here returns a MAP estimate: the single most probable f. This one
samples the posterior pi(f) ~ exp(-D(Hf, g) / beta) by Metropolis-Hastings and averages the
accepted samples, giving the posterior MEAN:

    f_MMSE  ~=  (1/T) sum_t f_t        over accepted samples

The integral defining that mean is intractable, but by the law of large numbers the
empirical average of accepted samples converges to it. The practical difference: a MAP
estimate can sit on a sharp, unrepresentative peak, while the MMSE average is smoother and
comes with a usable acceptance rate to judge whether the chain actually explored.

One proposal: data consistency -> noise -> positivity -> denoising. The current iterate is
corrected toward the data with a ridge inverse of the residual, perturbed, then denoised.

(v1 also offered a scheme "C" — noise BEFORE the data step — through a `proposal_method`
switch. It was removed: on MA-TIRF, with and without denoisers, both schemes gave the same
reconstructions to three decimals, so the switch was a letter with no effect. The order kept
is v1's default "D", the coherent one: the denoiser removes noise at the level just injected.)
THAT is what made MCMC problem-specific in v1: MA-TIRF inverted a dense matrix while
deconvolution applied a Wiener filter in Fourier — two files, 168 lines, one algorithm.
`ForwardOperator.ridge_inverse` is precisely that operation, defined for every operator, so
the two collapse into this one file.

CAUTION — beta is relative to the SCALED likelihood. D now carries the noise level
(solvers/fidelities/base.py): with Gaussian noise of variance b it is v1's D divided by b.
The same beta is therefore b times more selective than in v1, and a beta tuned before the
noise model existed must be multiplied by 1/b to reproduce the old acceptance rate. In
return beta reads as a genuine temperature: D is a per-pixel negative log-likelihood, so the
exact posterior of the Bayesian model corresponds to beta = 1 / N_g.

CAUTION — denoiser scale. Unlike PnP, this solver passes the raw tensor to the denoiser,
with a sigma (default 0.05) on f's own scale that also serves as the proposal noise level.
That is inconsistent with the project-wide [0, 255] convention (see solvers/denoising.py),
but it is v1's behaviour and changing it would silently alter every MCMC result. It is
preserved deliberately; revisit it only with a deliberate re-tuning of sigma.
"""

import time

import torch

from solvers.base import Solver
from solvers.fidelities import NOISE_MODEL_NAMES
from solvers.denoising import resolve, warn_if_slice_by_slice, NO_DENOISER
from solvers.objective_params import INIT_UI_PARAM
from solvers.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS
from core.features import Feature


MCMC_UI_PARAMS = {
    "max_iter": {
        "title": "Number of MCMC iterations (T)",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "T", "default": 200},
    },
    "beta": {
        "title": "Temperature (acceptance selectivity)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\beta", "default": 0.01},
    },
    "sigma": {
        "title": "Noise standard deviation (proposal + denoiser)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma", "default": 0.05},
    },
    "lambda_rr": {
        "title": "Ridge regularization (data consistency)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\lambda_{rr}",
                       "default": 10000.0},
    },
    "K": {
        "title": "Log every K iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "K", "default": 10},
    },
    "denoiser": {
        "title": "Denoiser (implicit prior)",
        "type": "option",
        "param_info": {"options_list": DENOISER_LIST},
    },
    "delta": {
        "title": "Anisotropy ratio coefficient",
        "type": "value",
        "requires": {Feature.ANISOTROPIC},
        "depends_on": {"denoiser": ANISOTROPIC_DENOISERS},
        "param_info": {"dtype": float, "unit": "",
                       "latex_name": "\\delta = \\frac{\\Delta z}{\\Delta xy}",
                       "default": 1.0},
    },
    **INIT_UI_PARAM,
}


class Mcmc(Solver):
    """Metropolis-Hastings posterior sampling, returning the mean of the accepted samples."""

    name = "MCMC"
    estimator_type = "MMSE"
    uses_denoiser = True
    supported_noise_models = NOISE_MODEL_NAMES   # the Metropolis ratio is computed from D itself
    uses_regularization = False
    ui_params = MCMC_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        max_iter = int(params.get("max_iter", 200))
        beta = float(params.get("beta", 0.01))
        sigma = float(params.get("sigma", 0.05))
        K = max(1, int(params.get("K", 10)))
        lambda_rr = float(params.get("lambda_rr", 10000.0))
        name = params.get("denoiser", NO_DENOISER)
        denoiser_fn = resolve(name)

        f = f0.clone()
        running_sum = torch.zeros_like(f)
        accepted = 0

        self.report(f"MCMC | T={max_iter} | sigma={sigma} | beta={beta} | "
                    f"denoiser={name}")
        warn_if_slice_by_slice(self.report, name, f)
        started = time.time()

        for k in range(1, max_iter + 1):
            if self.interrupted:
                self.report(f"Interrupted at iteration {k}.")
                break

            candidate = self._propose(objective, f, sigma, lambda_rr, denoiser_fn)
            if self._accept(objective, f, candidate, beta):
                f = candidate
                accepted += 1
                running_sum = running_sum + f

            if k % K:
                continue
            rate = accepted / k * 100
            self.report(f"iter {k:4d} | accepted={accepted}/{k} ({rate:.0f}%) | "
                        f"D={objective.data_term(f).item():.3e}")
            ## the live preview shows the running MMSE estimate, not the last sample:
            self.publish(running_sum / accepted if accepted else f)

        rate = accepted / max_iter * 100
        self.report(f"Accepted {accepted}/{max_iter} ({rate:.1f}%).")
        if accepted == 0:
            self.report("No sample was accepted — try a larger beta or a smaller sigma.")
        self.report(f"Execution in {time.time() - started:.2f} s.")
        return running_sum / accepted if accepted else f

    # ── the chain ────────────────────────────────────────────────────────────

    def _propose(self, objective, f, sigma, lambda_rr, denoiser_fn):
        """
        Build a candidate from the current iterate: a pull toward data consistency, then a
        random perturbation, then positivity and the denoising.
        """
        operator = objective.operator
        z = f + operator.ridge_inverse(objective.g - operator.apply(f), lambda_rr)
        z = z + sigma * torch.randn_like(z)
        z = z.clamp(min=0.0)
        if denoiser_fn is None:
            return z
        ## deliberately NOT on the [0, 255] scale — see the module docstring:
        return denoiser_fn(z, sigma).reshape(z.shape)

    @staticmethod
    def _accept(objective, f, candidate, beta) -> bool:
        """
        Metropolis-Hastings acceptance on the data fidelity:  a = min(1, exp((D_f - D_z)/beta)).

        A candidate that explains the data better is always accepted; one that explains it
        worse is accepted with a probability set by beta. That occasional acceptance of a
        worse sample is the whole point — it is what lets the chain explore the posterior
        instead of descending into the nearest minimum.
        """
        improvement = (objective.data_term(f) - objective.data_term(candidate)) / beta
        probability = torch.min(torch.tensor(1.0), torch.exp(improvement))
        return torch.rand(1).item() <= probability.item()
