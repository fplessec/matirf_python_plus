"""
PPXA — Parallel Proximal Algorithm, a MAP solver by operator splitting.

Minimizes  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)  subject to f >= 0, by
splitting it into three proximal sub-problems solved in parallel each iteration:

    p0 = prox of the data fidelity term        (a regularized normal solve)
    p1 = prox of the regularization term       (delegated to the Regularization object)
    p2 = projection onto f >= 0                (a clamp)

then recombining them by a weighted average with relaxation:

    p      = sum_i w_i p_i
    u_i   <- u_i + lambda_relax * (2p - f - p_i)
    f     <- f + lambda_relax * (p - f)

The relaxation lambda_relax is kept FIXED, in (0, 2) — 1.5 by default.

CHANGE FROM V1: v1 halved the relaxation each time the loss went back up. Two reasons to stop:
    > theory — PPXA converges for any relaxations in (0, 2) with sum rho_n (2 - rho_n) = inf
      (Combettes & Pesquet, 2008). A constant satisfies it; a geometric decay that keeps
      firing makes the sum finite and can stop the iteration short of the solution.
    > PPXA is not a descent method: it iterates on auxiliary variables, and the loss of the
      current average may rise while the iteration converges. The halving reacted to that
      normal behaviour. Measured on MA-TIRF (TV, SHV, Tikhonov; gamma 0.01 and 0.05; 1500
      iterations): it fired on one early transient and slowed every remaining iteration —
      e.g. SHV ended 7 % higher in L than with a fixed relaxation. Fixed 1.5 never let the
      loss rise and ended within 1-2 % of the best; fixed 1.9 reached the same L but could
      oscillate.
The run stops when the loss changes by less than EPS in either direction.

In v1 the data prox was the one thing PPXA could not do generically — it required inverting
(I + gamma(1-lambda) H^T H), so MA-TIRF had to subclass it. `ForwardOperator.solve_normal`
provides exactly that for every operator, so this file is now the whole of PPXA.
"""

import time

import torch

from solvers.base import Solver
from solvers.objective_params import INIT_UI_PARAM


PPXA_UI_PARAMS = {
    "max_iter": {
        "title": "Maximum iteration number",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{max\\_iter}",
                       "default": 2000},
    },
    "lambda_relax": {
        "title": "Relaxation parameter",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\lambda_{relax}",
                       "default": 1.5},
    },
    "gamma": {
        "title": "Data term step size",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\gamma",
                       "default": 0.05},
    },
    "K": {
        "title": "K check",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "K", "default": 10},
    },
    "EPS": {
        "title": "Stopping criterion",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\epsilon",
                       "default": 1e-8},
    },
    **INIT_UI_PARAM,
}


class Ppxa(Solver):
    """PPXA: three parallel proximal steps (data, prior, positivity) averaged with relaxation."""

    name = "PPXA"
    estimator_type = "MAP"
    supported_noise_models = frozenset({"gaussian"})   # the data step is a least-squares solve
    uses_regularization = True
    ui_params = PPXA_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        max_iter = int(params.get("max_iter", 2000))
        relax = float(params.get("lambda_relax", 1.5))
        if not 0.0 < relax < 2.0:
            raise ValueError(f"lambda_relax must lie in (0, 2) for PPXA to converge, got {relax}")
        gamma = float(params.get("gamma", 0.05))
        K = max(1, int(params.get("K", 10)))
        EPS = float(params.get("EPS", 1e-8))

        operator = objective.operator
        ## PPXA runs on L rescaled so that its data term reads 1/2 ||Hf - g||^2 — the scale
        ## gamma was tuned for in v1, before D carried the noise level. Rescaling L does not
        ## move its minimizer; it only keeps gamma's meaning. With w = quadratic_weight:
        ##     L / w = 1/2 ||Hf - g||^2 + (lambda / w) R
        ## so the data prox is argmin_p 1/2||p-u||^2 + gamma/2 ||Hp-g||^2, and the prior's
        ## prox is that of (gamma / w) * lambda * R. (w = 0 means lambda = 1: no data at all.)
        w = objective.quadratic_weight
        c = gamma if w > 0 else 0.0
        reg_tau = gamma / w if w > 0 else gamma
        Htg = operator.adjoint(objective.g)

        f = f0.clone()
        p = [torch.zeros_like(f) for _ in range(3)]
        u = [f.clone() for _ in range(3)]
        ## uniform weights over the three proximal terms:
        weights = torch.full((3,), 1.0 / 3.0, dtype=f.dtype, device=f.device)

        started = time.time()
        self.report(objective.describe())
        previous_loss = objective.value(f)
        self.report(f"iter    0 | loss={previous_loss.item():.3e} | relax={relax:.2e}")

        for iteration in range(1, max_iter + 1):
            if self.interrupted:
                self.report(f"Interrupted at iteration {iteration}.")
                return f.clamp(min=0.0)

            p[0] = self._data_prox(operator, u[0], Htg, c)
            p[1] = objective.prox_reg(u[1], reg_tau)
            p[2] = u[2].clamp(min=0.0)

            f, u = self._combine(p, u, f, weights, relax)

            if iteration % K:
                continue

            loss = objective.value(f.clamp(min=0.0))
            delta = (loss - previous_loss).item()
            self.report(f"iter {iteration:4d} | loss={loss.item():.3e} | "
                        f"relax={relax:.2e} | dloss={delta:+.3e}")
            self.publish(f.clamp(min=0.0))

            ## PPXA's loss is not monotone: stop on a small change of either sign
            if abs(delta) < EPS:
                self.report(f"Stopping criterion EPS reached after {iteration} iterations.")
                return self._finish(f, started)
            previous_loss = loss

        self.report(f"Maximum iteration number ({max_iter}) reached.")
        return self._finish(f, started)

    @staticmethod
    def _data_prox(operator, u, Htg, c):
        """
        (I + c H^T H)^-1 (u + c H^T g).

        Rewritten for solve_normal, which handles (H^T H + lam I)^-1:
            (I + cA)^-1 w = (1/c) (A + (1/c) I)^-1 w
        """
        if c <= 0:                          # lambda_reg == 1: the data term is switched off
            return u.clone()
        return operator.solve_normal(u + c * Htg, 1.0 / c) / c

    @staticmethod
    def _combine(p, u, f, weights, relax):
        """The PPXA recombination: weighted average of the proximal points, then relaxation."""
        averaged = torch.zeros_like(f)
        for weight, p_i in zip(weights, p):
            averaged = averaged + weight * p_i
        u = [u_i + relax * (2.0 * averaged - f - p_i) for u_i, p_i in zip(u, p)]
        f = f + relax * (averaged - f)
        return f, u

    def _finish(self, f, started):
        self.report(f"Execution in {time.time() - started:.2f} s.")
        return f.clamp(min=0.0)
