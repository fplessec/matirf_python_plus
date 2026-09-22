"""
PNPv2 — a PROPOSED alternative to PnP (solvers/pnp.py), kept as a separate solver.

Same algorithm: Half-Quadratic Splitting with the annealing schedule of Zhang et al. (DPIR),

    f <- (H^T H + alpha_k I)^-1 (H^T g + alpha_k z)      data
    z <- D_{sigma_k}(f)                                  denoiser (the prior)
    alpha_k sigma_k^2 = lambda_kz,   sigma_k: sigma -> sigma_final (log-spaced)

with the four changes proposed in docs/algorithms/pnp.md. Whether to keep it is the project
owner's decision, informed by the benchmark, where both are compared.

DIFFERENCES FROM PNP:
    > SCALE-FREE DENOISING. PnP feeds the denoisers 255 f, which assumes f in [0, 1] — true
      for a photograph, false for MA-TIRF (peak ~0.02). Measured at sigma = 5: Wiener changes
      the image by 84 % at MA-TIRF's scale and 20 % at a photograph's, DCT erases it. Here f
      is normalized by the peak p of the starting estimate: z = p D(255 f / p, sigma) / 255.
      sigma then means "noise level for an image whose peak is 255" on every problem, which
      is what makes a value transferable from one inverse problem to another.
    > NOISE-AWARE FINAL LEVEL. PnP ends the schedule at a hard-coded sigma_final = 1, i.e.
      always assumes a noise of 1/255 of the peak. Here sigma_final is the measurement's
      noise level relative to its peak, 255 x std(noise) / max(g), estimated from g
      (core.noise.estimate) — the DPIR choice, made scale-free. Can be set by hand.
    > ESTIMATED DELTA. Left empty, the anisotropy ratio of the Gaussian / Bilateral kernels is
      the operator's own estimate (as for Adam/PPXA's regularizations), not 1.
    > A PROPER START. PnP starts from H^T g (~850x too large on MA-TIRF), and the data step
      keeps the start's components in every direction H does not see. Here the start is the
      ridge estimate (H^T H + lambda_rr I)^-1 H^T g, with lambda_rr = s2^2 by default — the
      second EIGENVALUE of H^T H, the quantity lambda_rr is compared to: it keeps the best
      determined direction fully and the second half-way, and scales with the operator's
      gain (on MA-TIRF s2^2 = 36, inside the [s2, s1] = [6, 39] range the MCMC study found
      good). Falls back on s1^2 when the operator does not expose its spectrum. 'adjoint'
      remains available.

    Also: the schedule and the positivity are always on (PnP's `kai_zhang` and `forced_pos`
    switches had one sensible value). The denoisers offered are PnP's, DCT included (it is
    not used in the benchmark: its implementation is to be redone).

Parameters — critical: `denoiser`, `sigma`, `lambda_kz`. Fixed by rule (empty = automatic):
`sigma_final`, `delta`, `lambda_rr`. Comfort: `iter`, `init`.
"""

import time

import numpy as np
import torch

from core import noise
from core.features import Feature
from solvers.base import Solver
from solvers.denoising import resolve, denoise, warn_if_slice_by_slice
from solvers.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS

PNP_V2_UI_PARAMS = {
    "denoiser": {
        "title": "Denoiser (the prior)",
        "type": "option",
        "param_info": {"options_list": DENOISER_LIST},
    },
    "sigma": {
        "title": "Initial denoising level (0-255, relative to the image peak)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma", "default": 25.0},
    },
    "lambda_kz": {
        "title": "Final data weight",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\lambda_{kz}",
                       "default": 0.23},
    },
    "sigma_final": {
        "title": "Final denoising level (empty = the measurement's noise level)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\sigma_{final}",
                       "default": None},
    },
    "iter": {
        "title": "Number of annealing steps",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{iter}", "default": 16},
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
## below this final level (on the 0-255 scale) the last denoising is a no-op anyway
_SIGMA_FINAL_FLOOR = 0.5


class PnpV2(Solver):
    """Plug-and-Play HQS with scale-free denoising, a noise-aware schedule and a ridge start."""

    name = "PNPv2"
    estimator_type = "MAP"
    uses_denoiser = True
    supported_noise_models = frozenset({"gaussian"})   # the data step is a least-squares solve
    uses_regularization = False
    ui_params = PNP_V2_UI_PARAMS

    def initial_guess(self, objective, params):
        operator, g = objective.operator, objective.g
        if params.get("init", "ridge") == "adjoint":
            return operator.adjoint(g).detach().clamp(min=0.0)
        weight = params.get("lambda_rr")
        weight = self.second_eigenvalue(objective) if weight in _UNSET else float(weight)
        return operator.ridge_inverse(g, weight).detach().clamp(min=0.0)

    @staticmethod
    def second_eigenvalue(objective) -> float:
        """s2^2 (of H^T H) when the operator exposes its spectrum, else s1^2 (its norm)."""
        spectrum = getattr(objective.operator, "singular_values", None)
        if spectrum is not None:
            values = spectrum()
            if values.numel() > 1:
                return float(values[1]) ** 2
        like = objective.operator.adjoint(objective.g)
        return float(objective.operator.lipschitz(like))

    def solve(self, objective, f0, params):
        self.fix_randomness()

        n_iter = max(1, int(params.get("iter", 16)))
        sigma = float(params.get("sigma", 25.0))
        lambda_kz = float(params.get("lambda_kz", 0.23))
        name = params.get("denoiser", "None")
        denoiser_fn = resolve(name)
        operator, g = objective.operator, objective.g
        Htg = operator.adjoint(g)

        delta = params.get("delta")
        if delta in _UNSET:
            estimate = getattr(operator, "estimate_anisotropy_ratio", None)
            delta = estimate() if estimate is not None else 1.0
        delta = float(delta)

        sigma_final = params.get("sigma_final")
        sigma_final = (self.noise_level(g) if sigma_final in _UNSET else float(sigma_final))
        sigma_final = max(sigma_final, _SIGMA_FINAL_FLOOR)

        z = f0.clamp(min=0.0)
        ## the scale every denoising happens at: the start's peak
        peak = float(z.max()) or 1.0
        alphas, sigmas = self._schedule(sigma, sigma_final, lambda_kz, n_iter)
        warn_if_slice_by_slice(self.report, name, z)

        self.report(f"PNPv2 | denoiser={name} | sigma {sigma:g} -> {sigma_final:.3g} "
                    f"(relative to the peak {peak:.3g}) | lambda_kz={lambda_kz:g} | "
                    f"delta={delta:.3g}")
        started = time.time()
        f = z
        for k in range(n_iter):
            if self.interrupted:
                self.report(f"Interrupted at iteration {k + 1}.")
                return f

            f = operator.solve_normal(Htg + alphas[k] * z, alphas[k]).clamp(min=0.0)
            z = (peak * denoise(denoiser_fn, f / peak, sigmas[k], delta)).clamp(min=0.0)

            self.report(f"[PnPv2] iter {k + 1}/{n_iter} | alpha={alphas[k]:.3g} | "
                        f"sigma={sigmas[k]:.3g}")
            self.publish(f)

        self.report(f"Execution in {time.time() - started:.2f} s.")
        return f

    @staticmethod
    def noise_level(g: torch.Tensor) -> float:
        """The measurement's Gaussian noise std relative to its peak, on the 0-255 scale."""
        try:
            _, b = noise.estimate(g, poisson=False, gaussian=True)
        except ValueError:                           # too small to estimate: assume clean
            return 0.0
        peak = float(g.max()) or 1.0
        return 255.0 * (b ** 0.5) / peak

    @staticmethod
    def _schedule(sigma, sigma_final, lambda_kz, n_iter):
        """DPIR: sigma_k log-spaced from sigma down to sigma_final, alpha_k = lambda_kz / sigma_k^2."""
        sigma = max(sigma, sigma_final)
        sigmas = np.logspace(np.log10(sigma), np.log10(sigma_final), n_iter)
        alphas = lambda_kz * (sigma_final / sigmas) ** 2
        return alphas, sigmas


# ── ADMM-PnPv2 ────────────────────────────────────────────────────────────────

PNP_ADMM_V2_UI_PARAMS = {
    "denoiser": PNP_V2_UI_PARAMS["denoiser"],
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
    "init": PNP_V2_UI_PARAMS["init"],
    "lambda_rr": PNP_V2_UI_PARAMS["lambda_rr"],
    "delta": PNP_V2_UI_PARAMS["delta"],
}

## sigma left empty = this multiple of the measurement's noise level (to be settled by the
## benchmark: a first guess, not a result)
_NOISE_FACTOR = 3.0
_SIGMA_FLOOR = 1.0


class PnpAdmmV2(PnpV2):
    """
    ADMM-PnPv2 — a PROPOSED alternative to ADMM-PnP (solvers/pnp.py), kept separate.

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

    name = "ADMM-PnPv2"
    ui_params = PNP_ADMM_V2_UI_PARAMS

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
        sigma = (_NOISE_FACTOR * self.noise_level(g) if sigma in _UNSET else float(sigma))
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
