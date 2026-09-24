"""
ADMMv2 — a PROPOSED alternative to ADMM (solvers/admm.py), kept as a separate solver.

Same splitting, same prior (sparsity + positivity), same data step. Two changes, both
derived in docs/algorithms/admm.md; whether to keep this version is the project owner's
decision, informed by the benchmark, where both are compared.

Problem solved:

    min_{f >= 0}  1/2 ||H f - g||^2  +  tau ||f||_1,        tau = kappa * max(H^T g)

One iteration, with u = f the splitting constraint and eta the scaled dual variable:

    u   = (H^T H + mu I)^-1 (H^T g + mu (f - eta))        data step
    f   = max(u + eta - tau / mu, 0)                      soft threshold at tau / mu
    eta = eta + (u - f)                                   standard dual step

DIFFERENCES FROM ADMM:
    > the threshold is tau / mu with tau fixed by kappa, as in textbook ADMM. In ADMM the
      threshold is fixed, so the prior's weight at convergence is mu x threshold: changing
      mu changed the solution. Here mu changes only the speed.
    > the dual step is the standard eta += (u - f). ADMM multiplies it by mu, which makes mu
      the dual step size too and caps it at (1 + sqrt 5) / 2 ~ 1.618 for convergence.
    > kappa is a fraction of tau_max = max(H^T g), the smallest weight for which the
      solution is exactly zero (with positivity, the optimality condition H^T g <= tau at
      f = 0). So kappa = 1 gives an empty image, kappa -> 0 non-negative least squares, and
      the value in between reads as "how far towards the empty image" — independent of the
      data's scale, of H's gain and of the normalization of g.

Parameters:
    kappa     CRITICAL — the only prior knob: how sparse. Predicted useful range 0.001-0.1
              on MA-TIRF (to be established by the benchmark).
    mu        speed only. Keeps the singular directions with s^2 >> mu at each data step;
              predicted fastest in [s3^2, s1^2] (MA-TIRF: ~0.3 to ~1500).
    iter      budget; the iteration converges to the solution above.
"""

import time

import torch

from solvers.base import Solver


ADMM_V2_UI_PARAMS = {
    "kappa": {
        "title": "Sparsity (fraction of the weight that empties the image)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\kappa", "default": 0.01},
    },
    "mu": {
        "title": "Penalty (speed only)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\mu", "default": 1.0},
    },
    "iter": {
        "title": "Number of iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{iter}",
                       "default": 100},
    },
}


class AdmmV2(Solver):
    """ADMM for non-negative lasso: data step, threshold tau / mu, standard dual step."""

    name = "ADMMv2"
    estimator_type = "MAP"
    supported_noise_models = frozenset({"gaussian"})   # the data step is a least-squares solve
    uses_regularization = False
    ui_params = ADMM_V2_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        n_iter = int(params.get("iter", 100))
        mu = float(params.get("mu", 1.0))
        kappa = float(params.get("kappa", 0.01))
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")

        operator, g = objective.operator, objective.g
        Htg = operator.adjoint(g)
        tau = kappa * float(Htg.max().clamp(min=0))       # tau_max = max(H^T g)
        threshold = tau / mu

        ## start from the data step itself, with f = eta = 0
        f = operator.solve_normal(Htg, mu).clamp(min=0.0)
        eta = torch.zeros_like(f)
        zero = torch.zeros_like(f)

        started = time.time()
        self.report(f"ADMMv2 | kappa={kappa:.3g} (tau={tau:.3e}) | mu={mu:.3g}")

        for iteration in range(1, n_iter + 1):
            if self.interrupted:
                self.report(f"Interrupted at iteration {iteration}.")
                return f

            u = operator.solve_normal(Htg + mu * (f - eta), mu)
            f = torch.max(u + eta - threshold, zero)
            eta = eta + (u - f)

            if iteration % 10 == 0 or iteration == n_iter:
                gap = float((u - f).norm() / f.norm().clamp(min=1e-30))
                self.report(f"iter {iteration:4d}/{n_iter} | ||u - f|| / ||f|| = {gap:.2e}")
                self.publish(f)

        self.report(f"Execution in {time.time() - started:.2f} s.")
        return f
