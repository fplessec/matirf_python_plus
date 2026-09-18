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

The relaxation factor is halved whenever the loss goes back up, which is what keeps an
aggressive lambda_relax (1.9 by default, near the theoretical limit of 2) safe.

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
                       "default": 1.9},
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
    uses_regularization = True
    ui_params = PPXA_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        max_iter = int(params.get("max_iter", 2000))
        relax = float(params.get("lambda_relax", 1.9))
        gamma = float(params.get("gamma", 0.05))
        K = max(1, int(params.get("K", 10)))
        EPS = float(params.get("EPS", 1e-8))

        operator = objective.operator
        ## the data prox is the quadratic (Gaussian) one: argmin_p 1/2||p-u||^2 + c/2||Hp-g||^2
        ## with c = gamma * (1 - lambda_reg), i.e. p = (I + c H^T H)^-1 (u + c H^T g).
        c = gamma * objective.data_weight
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
            p[1] = objective.prox_reg(u[1])
            p[2] = u[2].clamp(min=0.0)

            f, u = self._combine(p, u, f, weights, relax)

            if iteration % K:
                continue

            loss = objective.value(f.clamp(min=0.0))
            delta = (loss - previous_loss).item()
            self.report(f"iter {iteration:4d} | loss={loss.item():.3e} | "
                        f"relax={relax:.2e} | dloss={delta:+.3e}")
            self.publish(f.clamp(min=0.0))

            if delta > 0:
                relax = relax / 2          # overshooting: damp the relaxation
            elif delta > -EPS:
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
