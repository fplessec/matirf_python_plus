"""
ADMM-PnP — Plug-and-Play ADMM: a denoiser as the prior, with a dual variable.

The augmented-Lagrangian counterpart of PnP (solvers/pnp.py), with a dual variable u:

    x <- (H^T H + rho I)^-1 (H^T g + rho (v - u))    data
    v <- D_sigma(x + u)                              prior
    u <- u + (x - v)                                 dual

Keeping the dual lets rho stay FIXED — no annealing schedule to tune.

It does not consult an explicit regularization term: `uses_regularization = False`, and no
regularization parameters are offered. sigma is on the [0, 255] denoiser scale; see
solvers/denoising.py for why.

It was problem-specific in v1 solely because of its inversion step;
`ForwardOperator.solve_normal` supplies it generically, so nothing here is specialized.
"""

import time

import torch

from solvers.base import Solver
from solvers.denoising import resolve, denoise, warn_if_slice_by_slice, DENOISER_UI_PARAMS


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
    **DENOISER_UI_PARAMS,
}


class PnpAdmm(Solver):
    """Plug-and-Play ADMM: a denoiser prior with a dual variable, so rho stays fixed."""

    name = "ADMM-PnP"
    estimator_type = "MAP"
    uses_denoiser = True
    supported_noise_models = frozenset({"gaussian"})   # the data step is a least-squares solve
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
