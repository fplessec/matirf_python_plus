"""
MCMCv2 — a PROPOSED alternative to MCMC (solvers/mcmc.py), kept as a separate solver.

It is the same chain as MCMC, with its parameters made relative and calibrated after a
measured study (below). It does not replace MCMC: whether to adopt it is the project
owner's decision, informed by the benchmark (benchmarks/), where both are compared.

An MMSE estimator: the mean of a Markov chain of plausible reconstructions.

Every other solver here returns a MAP estimate, the single most probable f. This one runs a
Metropolis-Hastings chain and returns the MEAN of its states after a burn-in: an average of
many reconstructions that all explain the data, smoother and less committed to one sharp
optimum than a MAP estimate.

One step of the chain, from the current state f:

    z = f + (H^T H + lambda_rr I)^-1 H^T (g - H f)     pull towards the data
    z = z + sigma * noise                               explore
    z = denoise(max(z, 0), sigma)                        positivity, then the prior
    accept z with probability min(1, exp((D(f) - D(z)) / beta))

The denoiser IS the prior: it removes the noise that was just injected, at the same level
sigma, and what it keeps is what it considers image. Without a denoiser the chain has no
prior at all and cannot beat its own starting point.

--------------------------------------------------------------------------------------
The parameters — and what the study behind them found (see benchmarks/)
--------------------------------------------------------------------------------------

    denoiser    CRITICAL. The prior. TV Bregman is the most robust across noise levels;
                Wiener is excellent at low noise and degrades at high noise; DCT breaks
                the reconstruction (rejected by the realism check). 'None' = no prior.

    sigma       CRITICAL. Exploration step and denoising strength together, as a FRACTION
                OF THE IMAGE PEAK (of the starting estimate), so it does not depend on the
                data's scale. ~0.02-0.05 is good; its optimum grows with the noise.
                -> 0 : no exploration, no denoising — the chain stays at its start
                large : over-smoothing, fine structures erased

    lambda_rr   Ridge weight of the data step. Empty = automatic: the operator's largest
                singular value s1. Any value between s2 and s1 works equally well (the data
                step then keeps the directions the data determine and cuts the rest); below
                s3 it amplifies noise, far above s1 (e.g. 1e4) it barely pulls.

    temperature NOT critical. beta is calibrated automatically as `temperature` x the
                typical increase of D a proposal causes — measured on the first proposals.
                Why: acceptance is a SWITCH, not a dial. With beta below that typical
                increase nothing is accepted and the chain returns its start; above it
                almost everything is (90-99 %) and the result no longer depends on beta.
                Since that increase grows like sigma^2 and with the noise model, no fixed
                beta can work across settings — which is why v1's MCMC "worked with very
                specific parameters or not at all". Relative, 0.3 to 10 all give the same.

    max_iter    The chain converges in ~150 iterations; the first 25 % are discarded
                (burn-in) and the rest averaged. The average beats the last state clearly.

    init        Not critical: adjoint, ridge(1e4) and ridge(s2) starts reach the same mean.

DIFFERENCES FROM MCMC, all measured before being made:
    > beta and sigma are relative (see above), so values carry over between problems and
      noise levels; MCMC's absolute beta and sigma are not read.
    > lambda_rr defaults to the automatic s1.
    > the estimate is the mean of ALL states after burn-in (standard MCMC), not of the
      accepted proposals only.

`ForwardOperator.ridge_inverse` provides the data step for every operator, which is what
made this file problem-agnostic (v1 had one MCMC per problem).
"""

import time

import torch

from solvers.base import Solver, ridge_weight
from solvers.fidelities import NOISE_MODEL_NAMES
from solvers.denoising import resolve, warn_if_slice_by_slice, NO_DENOISER
from solvers.objective_params import INIT_UI_PARAM
from solvers.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS
from core.features import Feature


## proposals used to calibrate beta, and the fraction of the chain discarded as burn-in
_CALIBRATION_PROPOSALS = 5
_BURN_IN = 0.25


MCMC_V2_UI_PARAMS = {
    "denoiser": {
        "title": "Denoiser (the prior)",
        "type": "option",
        "param_info": {"options_list": DENOISER_LIST},
    },
    "sigma": {
        "title": "Exploration / denoising strength (fraction of the image peak)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma", "default": 0.03},
    },
    "lambda_rr": {
        "title": "Ridge weight of the data step (empty = automatic)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\lambda_{rr}",
                       "default": None},
    },
    "temperature": {
        "title": "Acceptance temperature (relative, not critical)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\tau", "default": 1.0},
    },
    "max_iter": {
        "title": "Number of MCMC iterations (T)",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "T", "default": 300},
    },
    "K": {
        "title": "Log every K iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "K", "default": 50},
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


class McmcV2(Solver):
    """Metropolis-Hastings posterior sampling, returning the mean of the accepted samples."""

    name = "MCMCv2"
    estimator_type = "MMSE"
    uses_denoiser = True
    supported_noise_models = NOISE_MODEL_NAMES   # the Metropolis ratio is computed from D itself
    uses_regularization = False
    ui_params = MCMC_V2_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        max_iter = int(params.get("max_iter", 300))
        K = max(1, int(params.get("K", 50)))
        name = params.get("denoiser", NO_DENOISER)
        denoiser_fn = resolve(name)
        lambda_rr = ridge_weight(objective, params.get("lambda_rr"))
        ## sigma is relative to the image's peak, so the same value suits any data scale
        peak = float(f0.clamp(min=0).max()) or 1.0
        sigma = float(params.get("sigma", 0.03)) * peak
        beta = self._calibrate(objective, f0, sigma, lambda_rr, denoiser_fn,
                               float(params.get("temperature", 1.0)))

        f = f0.clone()
        burn_in = int(_BURN_IN * max_iter)
        running_sum = torch.zeros_like(f)
        accepted = 0

        self.report(f"MCMC | T={max_iter} (burn-in {burn_in}) | denoiser={name} | "
                    f"sigma={params.get('sigma', 0.03)} x peak | lambda_rr={lambda_rr:.3g} | "
                    f"beta={beta:.3g} (calibrated)")
        warn_if_slice_by_slice(self.report, name, f)
        started = time.time()
        k = 0

        for k in range(1, max_iter + 1):
            if self.interrupted:
                self.report(f"Interrupted at iteration {k}.")
                break

            candidate = self._propose(objective, f, sigma, lambda_rr, denoiser_fn)
            if self._accept(objective, f, candidate, beta):
                f = candidate
                accepted += 1
            if k > burn_in:
                running_sum = running_sum + f

            if k % K:
                continue
            self.report(f"iter {k:4d} | accepted={accepted}/{k} ({accepted / k * 100:.0f}%) | "
                        f"D={objective.data_term(f).item():.3e}")
            ## the live preview shows the running MMSE estimate once the burn-in is over:
            self.publish(running_sum / (k - burn_in) if k > burn_in else f)

        kept = max(1, min(k, max_iter) - burn_in)
        self.report(f"Accepted {accepted}/{max_iter} ({accepted / max_iter * 100:.1f}%).")
        if accepted == 0:
            self.report("No proposal was accepted: raise the temperature.")
        self.report(f"Execution in {time.time() - started:.2f} s.")
        return running_sum / kept if k > burn_in else f

    # ── the chain ────────────────────────────────────────────────────────────

    def _calibrate(self, objective, f0, sigma, lambda_rr, denoiser_fn, temperature) -> float:
        """beta = temperature x the typical increase of D caused by one proposal from f0."""
        start = objective.data_term(f0)
        increases = [float(objective.data_term(
            self._propose(objective, f0, sigma, lambda_rr, denoiser_fn)) - start)
            for _ in range(_CALIBRATION_PROPOSALS)]
        typical = float(torch.tensor(increases).abs().median())
        return temperature * max(typical, 1e-12)

    def _propose(self, objective, f, sigma, lambda_rr, denoiser_fn):
        """Pull towards the data, explore, then positivity and the prior (the denoiser)."""
        operator = objective.operator
        z = f + operator.ridge_inverse(objective.g - operator.apply(f), lambda_rr)
        z = (z + sigma * torch.randn_like(z)).clamp(min=0.0)
        if denoiser_fn is None:
            return z
        ## on f's own scale, at the level of the noise just injected (see the module doc)
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
