"""
PnP — Plug-and-Play reconstruction whose prior is a denoiser rather than a formula.

Replace the proximal operator of an explicit prior R(f) by a call to an off-the-shelf
denoiser (Venkatakrishnan et al., 2013). The prior is then whatever the denoiser implicitly
encodes — usually far richer than a hand-written R(f) — at the cost of losing the
closed-form objective.

This solver is Half-Quadratic Splitting: no dual variable.

    f <- (H^T H + alpha I)^-1 (H^T g + alpha z)      inversion
    z <- D_sigma(f)                                  denoising

It supports Zhang et al. (DPIR) scheduling: alpha grows and sigma shrinks logarithmically
across iterations, which is what makes plain HQS converge. Its augmented-Lagrangian
counterpart, with a dual variable, is ADMM-PnP (solvers/pnp_admm.py).

It does not consult an explicit regularization term: `uses_regularization = False`, and no
regularization parameters are offered. sigma is on the [0, 255] denoiser scale; see
solvers/denoising.py for why.

It was problem-specific in v1 solely because of its inversion step;
`ForwardOperator.solve_normal` supplies it generically, so nothing here is specialized.
"""

import time

import numpy as np
import torch

from solvers.base import Solver
from solvers.denoising import resolve, denoise, warn_if_slice_by_slice, DENOISER_UI_PARAMS




PNP_UI_PARAMS = {
    "iter": {
        "title": "Number of PnP iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{iter}", "default": 5},
    },
    "sigma": {
        "title": "Noise standard deviation (0-255 scale)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma", "default": 5},
    },
    "kai_zhang": {
        "title": "Use Kai Zhang scheduling",
        "type": "bool",
        "param_info": {"default": True},
    },
    "lambda_kz": {
        "title": "Kai Zhang coefficient",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\lambda_{kz}",
                       "default": 0.23},
    },
    **DENOISER_UI_PARAMS,
}


class Pnp(Solver):
    """Plug-and-Play HQS: alternate a data inversion and a denoising step, with annealing."""

    name = "PNP"
    estimator_type = "MAP"
    uses_denoiser = True
    supported_noise_models = frozenset({"gaussian"})   # the data step is a least-squares solve
    uses_regularization = False
    ui_params = PNP_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        n_iter = int(params.get("iter", 5))
        sigma = float(params.get("sigma", 5))
        delta = float(params.get("delta", 1.0))
        lambda_kz = float(params.get("lambda_kz", 0.23))
        forced_pos = bool(params.get("forced_pos", True))
        name = params.get("denoiser", "None")

        operator = objective.operator
        Htg = operator.adjoint(objective.g)
        denoiser_fn = resolve(name)

        z = f0.clamp(min=0.0)
        warn_if_slice_by_slice(self.report, name, z)
        alphas, sigmas = self._schedule(params.get("kai_zhang", True), sigma, lambda_kz, n_iter)

        started = time.time()
        f = z
        for k in range(n_iter):
            if self.interrupted:
                self.report(f"Interrupted at iteration {k + 1}.")
                return f

            self.report(f"[PnP] iter {k + 1}/{n_iter} | "
                        f"alpha={alphas[k]:.3g} | sigma={sigmas[k]:.3g}")

            f = operator.solve_normal(Htg + alphas[k] * z, alphas[k])   # inversion
            z = denoise(denoiser_fn, f, sigmas[k], delta)               # implicit prior

            if forced_pos:
                f = f.clamp(min=0.0)
                z = z.clamp(min=0.0)
            self.publish(f)

        self.report(f"Execution in {time.time() - started:.2f} s.")
        return f

    @staticmethod
    def _schedule(kai_zhang, sigma, lambda_kz, n_iter):
        """
        Zhang et al. (DPIR) annealing: alpha rises and sigma falls logarithmically.

        Starting with a weak data term and strong denoising, then reversing, is what lets
        plain HQS converge without a dual variable. Disabled -> both stay constant.
        """
        if not kai_zhang:
            return [lambda_kz] * n_iter, [sigma] * n_iter
        sigma_final = 1.0
        alpha_first = lambda_kz * sigma_final ** 2 / sigma ** 2
        alphas = np.logspace(np.log10(alpha_first), np.log10(lambda_kz), n_iter)
        sigmas = np.sqrt(lambda_kz / alphas) * sigma_final
        return alphas, sigmas
