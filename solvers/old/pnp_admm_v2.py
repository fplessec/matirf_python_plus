"""ADMM-PnPv2 — a PROPOSED alternative to ADMM-PnP (solvers/pnp_admm.py), kept separate.

Same algorithm (scaled ADMM with a denoiser as the prior's proximal step, standard dual
update, fixed rho and sigma):

    x <- (H^T H + rho I)^-1 (H^T g + rho (v - u))       data
    v <- max(D_sigma(x + u), 0)                         denoiser (the prior)
    u <- u + (x - v)                                    dual

with the proposals of docs/algorithms/admm_pnp.md §4:
    > scale-free denoising (v = p D(255 (x + u) / p, sigma) / 255, p the start's peak), as
      PNPv2 — sigma transfers between inverse problems;
    > estimated delta when left empty;
    > a ridge start, lambda_rr = s2^2 by default (the second eigenvalue of H^T H), or
      'adjoint';
    > sigma tied to the noise when left empty: 3 x the measurement's noise level relative
      to its peak (0-255 scale). With no schedule, sigma is ADMM-PnP's only prior knob;
      tying it to the noise lets the prior follow the data. The factor 3 is a first guess
      for the benchmark to settle.
Positivity is always on. Not changed (open point): rho is still absolute, compared to the
eigenvalues of H^T H (see docs/algorithms/pnp.md §5).

Parameters — critical: `denoiser`, `sigma`, `rho` (prior strength ~ rho sigma^2).
Fixed by rule (empty = automatic): `sigma`, `delta`, `lambda_rr`. Comfort: `iter`, `init`.
"""

import time

import torch

from core.features import Feature
from solvers.base import Solver, ridge_start
from solvers.denoising import resolve, denoise, warn_if_slice_by_slice, relative_noise_level
from solvers.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS


PNP_ADMM_V2_UI_PARAMS = {
    "denoiser": {
        "title": "Denoiser (the prior)",
        "type": "option",
        "param_info": {"options_list": DENOISER_LIST},
    },
    "sigma": {
        "title": "Denoising level (0-255, relative to the peak; empty = 3 x the noise)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma", "default": None},
    },
    "rho": {
        "title": "Penalty (data-step ridge)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\rho", "default": 0.1},
    },
    "iter": {
        "title": "Number of iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{iter}", "default": 50},
    },
    "init": {
        "title": "Initialization",
        "type": "option",
        "param_info": {"options_list": ["ridge", "adjoint"]},
    },
    "lambda_rr": {
        "title": "Ridge weight of the start (empty = s2^2)",
        "type": "value",
        "depends_on": {"init": "ridge"},
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\lambda_{rr}",
                       "default": None},
    },
    "delta": {
        "title": "Anisotropy ratio (empty = estimated)",
        "type": "value",
        "requires": {Feature.ANISOTROPIC},
        "depends_on": {"denoiser": ANISOTROPIC_DENOISERS},
        "param_info": {"dtype": float, "unit": "",
                       "latex_name": "\\delta = \\frac{\\Delta z}{\\Delta xy}",
                       "default": None},
    },
}

_UNSET = (None, "", "None", "null", "auto")
## sigma left empty = this multiple of the measurement's noise level (to be settled by the
## benchmark: a first guess, not a result)
_NOISE_FACTOR = 3.0
_SIGMA_FLOOR = 1.0


class PnpAdmmV2(Solver):
    """Plug-and-Play ADMM with scale-free denoising, a noise-tied sigma and a ridge start."""

    name = "ADMM-PnPv2"
    estimator_type = "MAP"
    uses_denoiser = True
    supported_noise_models = frozenset({"gaussian"})   # the data step is a least-squares solve
    uses_regularization = False
    ui_params = PNP_ADMM_V2_UI_PARAMS

    def initial_guess(self, objective, params):
        return ridge_start(objective, params)

    def solve(self, objective, f0, params):
        self.fix_randomness()

        n_iter = max(1, int(params.get("iter", 50)))
        rho = float(params.get("rho", 0.1))
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        name = params.get("denoiser", "None")
        denoiser_fn = resolve(name)
        operator, g = objective.operator, objective.g
        Htg = operator.adjoint(g)

        delta = params.get("delta")
        if delta in _UNSET:
            estimate = getattr(operator, "estimate_anisotropy_ratio", None)
            delta = estimate() if estimate is not None else 1.0
        delta = float(delta)

        sigma = params.get("sigma")
        sigma = (_NOISE_FACTOR * relative_noise_level(g) if sigma in _UNSET else float(sigma))
        sigma = max(sigma, _SIGMA_FLOOR)

        x = f0.clamp(min=0.0)
        peak = float(x.max()) or 1.0
        v, u = x.clone(), torch.zeros_like(x)
        warn_if_slice_by_slice(self.report, name, x)

        self.report(f"ADMM-PnPv2 | denoiser={name} | sigma={sigma:.3g} (relative to the peak "
                    f"{peak:.3g}) | rho={rho:g} | delta={delta:.3g}")
        started = time.time()
        for k in range(n_iter):
            if self.interrupted:
                self.report(f"Interrupted at iteration {k + 1}.")
                return v

            x = operator.solve_normal(Htg + rho * (v - u), rho)
            v = (peak * denoise(denoiser_fn, (x + u) / peak, sigma, delta)).clamp(min=0.0)
            u = u + (x - v)

            if (k + 1) % 10 == 0 or k + 1 == n_iter:
                change = float((x - v).norm() / v.norm().clamp(min=1e-30))
                self.report(f"[ADMM-PnPv2] iter {k + 1}/{n_iter} | ||x - v|| / ||v|| = {change:.2e}")
                self.publish(v)

        self.report(f"Execution in {time.time() - started:.2f} s.")
        return v
