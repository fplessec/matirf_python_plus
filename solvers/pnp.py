"""
Plug-and-Play — reconstruction whose prior is a denoiser rather than a formula.

Two variants live here, sharing the same idea: replace the proximal operator of an explicit
prior R(f) by a call to an off-the-shelf denoiser (Venkatakrishnan et al., 2013). The prior
is then whatever the denoiser implicitly encodes — usually far richer than a hand-written
R(f) — at the cost of losing the closed-form objective.

    Pnp       Half-Quadratic Splitting: no dual variable.
                  f <- (H^T H + alpha I)^-1 (H^T g + alpha z)      inversion
                  z <- D_sigma(f)                                  denoising
              Supports Zhang et al. (DPIR) scheduling: alpha grows and sigma shrinks
              logarithmically across iterations, which is what makes plain HQS converge.

    PnpAdmm   The augmented-Lagrangian counterpart, with a dual variable u.
                  x <- (H^T H + rho I)^-1 (H^T g + rho (v - u))    data
                  v <- D_sigma(x + u)                              prior
                  u <- u + (x - v)                                 dual
              Keeping the dual lets rho stay FIXED — no annealing schedule to tune.

Because neither consults an explicit regularization term, both declare
`uses_regularization = False` and are offered no regularization parameters.

sigma is on the [0, 255] denoiser scale for both; see solvers/denoising.py for why.

Both were problem-specific in v1 solely because of their inversion step;
`ForwardOperator.solve_normal` supplies it generically, so nothing here is specialized.
"""

import time

import numpy as np
import torch

from solvers.base import Solver
from solvers.denoising import resolve, denoise, warn_if_slice_by_slice
from common.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS
from core.features import Feature


_DENOISER_PARAM = {
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
    "forced_pos": {
        "title": "Forced positivity",
        "type": "bool",
        "param_info": {"default": True},
    },
}


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
    **_DENOISER_PARAM,
}


PNP_ADMM_UI_PARAMS = {
    "iter": {
        "title": "Number of ADMM-PnP iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{iter}", "default": 20},
    },
    "rho": {
        "title": "Penalty coefficient",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\rho", "default": 1.0},
    },
    "sigma": {
        "title": "Denoiser strength (0-255 scale)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma", "default": 15},
    },
    **_DENOISER_PARAM,
}


class Pnp(Solver):
    """Plug-and-Play HQS: alternate a data inversion and a denoising step, with annealing."""

    name = "PNP"
    estimator_type = "MAP"
    uses_denoiser = True
    uses_data_fidelity = False
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


class PnpAdmm(Solver):
    """Plug-and-Play ADMM: a denoiser prior with a dual variable, so rho stays fixed."""

    name = "ADMM-PnP"
    estimator_type = "MAP"
    uses_denoiser = True
    uses_data_fidelity = False
    uses_regularization = False
    ui_params = PNP_ADMM_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        n_iter = int(params.get("iter", 20))
        rho = float(params.get("rho", 1.0))
        sigma = float(params.get("sigma", 15))
        delta = float(params.get("delta", 1.0))
        forced_pos = bool(params.get("forced_pos", True))
        name = params.get("denoiser", "None")

        operator = objective.operator
        Htg = operator.adjoint(objective.g)
        denoiser_fn = resolve(name)

        x = f0.clamp(min=0.0)
        warn_if_slice_by_slice(self.report, name, x)
        v = x.clone()
        u = torch.zeros_like(x)

        started = time.time()
        for k in range(n_iter):
            if self.interrupted:
                self.report(f"Interrupted at iteration {k + 1}.")
                return v

            self.report(f"[ADMM-PnP] iter {k + 1}/{n_iter} | rho={rho:.3g} | sigma={sigma:.3g}")

            x = operator.solve_normal(Htg + rho * (v - u), rho)   # data step
            v = denoise(denoiser_fn, x + u, sigma, delta)         # prior step
            if forced_pos:
                v = v.clamp(min=0.0)
            u = u + (x - v)                                       # dual step
            self.publish(v)

        self.report(f"Execution in {time.time() - started:.2f} s.")
        return v
