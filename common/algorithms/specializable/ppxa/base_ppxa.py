"""
Generic PPXA (Parallel Proximal Algorithm) for inverse problems.

Minimizes:  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)
            subject to f >= 0

PPXA splits the problem into proximal sub-problems solved in parallel:
    p[0] = prox of the data fidelity term
    p[1] = prox of the regularization term
    p[2] = prox of the positivity constraint (projection onto f >= 0)

Then updates f via weighted average and relaxation.

Subclasses must override:
    apply_forward(H, f)                : computes Hf
    apply_adjoint(H, x)               : computes H^t x
    init_f(g, H, params)              : initialization of f
    precompute(g, H, params)           : precomputes matrices/operators needed by prox steps
    compute_data_prox(u, params)       : proximal operator for the data fidelity term
"""

from typing import Dict, Any
import time

import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.ppxa.ui_params import PPXA_UI_PARAMETERS
from common.utils import get_variables_from_dict


class BasePpxa(Algorithm):

    name = "PPXA"
    ui_params = PPXA_UI_PARAMETERS
    estimator_type = "MAP"
    uses_denoiser = False
    uses_regularization = True

    def run(self, g, H, params: Dict[str, Any]):
        self.fix_randomness()

        (max_iter, lambda_relax, K, EPS) = get_variables_from_dict(
            params, ['max_iter', 'lambda_relax', 'K', 'EPS'])

        self.precompute(g, H, params)

        self._diff_ops = self._create_diff_ops(params)
        self._regularization = self._create_regularization(params)
        lambda_reg = params.get('lambda_reg', 0.)

        f = self.init_f(g, H, params)
        loss_computer = self._create_loss_computer(g, H, params)  # (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

        # PPXA initial terms: 3 proximal variables
        p = [torch.zeros_like(f) for _ in range(3)]
        u = [f.clone() for _ in range(3)]

        # weights for each 3 ppxa terms
        weights = self.init_weights(params)

        t0 = time.time()
        loss = loss_computer(f)
        self._print(f"iter 0 \nloss={loss:.3e} | lambda_relax={lambda_relax:.2e}")
        prev_loss = loss

        for it in range(max_iter):
            if self.is_stop_requested():
                return None

            # proximal steps (parallel):
            p[0] = self.compute_data_prox(u[0], params)
            p[1] = self._regularization.prox(u[1], lambda_reg, self._diff_ops)
            p[2] = u[2].clamp(min=0.)  # positivity constraint prox

            # PPXA inner step: weighted average + relaxation update
            f, u = self._ppxa_inner_step(p, u, f, weights, lambda_relax)

            if it % K == K - 1:
                loss = loss_computer(f.clone().clamp(min=0.))
                dloss = loss - prev_loss
                self._print(
                    f"iter {it + 1:4d} \nloss={loss.item():.3e} | "
                    f"lambda_relax={lambda_relax:.2e} | "
                    f"dloss={dloss:+.3e}"
                )
                self._update_figure(f.clone().clamp(min=0.))
                if dloss > 0:
                    lambda_relax = lambda_relax / 2
                elif dloss > -EPS:
                    self._print("\nThe stopping criterion EPS has been met.")
                    self._print(f"Execution in {time.time() - t0:.2f} s.")
                    return f.clamp(min=0.)
                prev_loss = loss

        self._print("\nThe maximum iterations number has been reached.")
        self._print(f"Execution in {time.time() - t0:.2f} s.")
        return f.clamp(min=0.)

    @staticmethod
    def _ppxa_inner_step(p, u, f, weights, lambda_relax=1.9):
        """
        Inner step of the ppxa algorithm using different weights
        p_{k} = \\sum w_i p_{i,k}
        u_{i,k+1} = u_{i,k} + \\lambda ( 2 p_{k} - x_k - p_{i,k} )
        x_{k+1} = x_k + \\lambda (p_k - x_k)
        """
        # compute the weighted sum of the proximal operators:
        pl = torch.zeros_like(f)
        for j in range(len(p)):
            pl += weights[j] * p[j]
        # updates the ppxa terms
        for j in range(len(u)):
            u[j] = u[j] + lambda_relax * (2.0 * pl - f - p[j])
        # update the solution:
        f += lambda_relax * (pl - f)
        return f, u

    def init_weights(self, params):
        """Returns the weight tensor for the 3 proximal terms. Default: uniform."""
        import common.settings as settings
        weights = torch.tensor([1., 1., 1.], dtype=settings.dtype, device=settings.device)
        weights /= weights.sum()
        return weights

    def init_f(self, g, H, params):
        """Initialization of f. Default: f0 = H^t g."""
        return self.apply_adjoint(H, g.clone())

    def precompute(self, g, H, params):
        """Precomputes matrices/operators needed by prox steps. Store as self attributes."""
        pass

    def compute_data_prox(self, u, params):
        """Proximal operator for the data fidelity term. Must be overridden."""
        raise NotImplementedError
