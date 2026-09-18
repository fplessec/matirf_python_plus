"""
ADMM — Alternating Direction Method of Multipliers with soft thresholding.

Solves  min_f  1/2 ||Hf - g||^2  +  a sparsity/positivity prior applied by thresholding,
through the splitting u = f and its augmented Lagrangian:

    u-step (data)   u = (H^T H + mu I)^-1 (H^T g + mu (f - eta))
    f-step (prior)  f = max(u + eta - threshold, 0)
    eta-step (dual) eta = eta + mu (u - f)

References: https://arxiv.org/pdf/1801.00882 and the Pol-TIRF implementation.

Unlike Adam and PPXA, this solver does not consult the objective's regularization term: its
prior is the fixed soft-threshold above, which is why `uses_regularization` is False and no
regularization parameters are offered for it.

BEHAVIOUR NOTE — the threshold. v1's MA-TIRF subclass computed it as
`g.max() / H.max() * 0.1`, which needs H as an explicit matrix and so cannot exist for an
operator applied in Fourier. v2 uses `threshold_ratio * g.max() / ||H||`, with ||H|| the
operator norm obtained from `lipschitz`. It is the same quantity in spirit (a fraction of
the measurement scale, divided by the operator's gain) and it is defined for every operator,
but for MA-TIRF it is NOT numerically identical to v1, since max-element and operator norm
differ. Tune `threshold_ratio` if a ported MA-TIRF run looks over- or under-sparse.
"""

import time

import torch

from solvers.base import Solver


ADMM_UI_PARAMS = {
    "iter": {
        "title": "Number of iterations",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{iter}",
                       "default": 20},
    },
    "mu": {
        "title": "Mu coefficient (penalty)",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\mu", "default": 0.5},
    },
    "threshold_ratio": {
        "title": "Soft-threshold ratio",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\kappa",
                       "default": 0.1},
    },
}


class Admm(Solver):
    """ADMM with a soft-thresholding prior: data step, thresholding step, dual update."""

    name = "ADMM"
    estimator_type = "MAP"
    uses_data_fidelity = False
    uses_regularization = False
    ui_params = ADMM_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        n_iter = int(params.get("iter", 20))
        mu = float(params.get("mu", 0.5))
        ratio = float(params.get("threshold_ratio", 0.1))

        operator, g = objective.operator, objective.g
        Htg = operator.adjoint(g)
        threshold = self._threshold(operator, g, f0, ratio)

        ## the natural start for ADMM is the data step itself, from f = eta = 0:
        f = operator.solve_normal(Htg, mu)
        eta = torch.zeros_like(f)
        zero = torch.zeros_like(f)

        started = time.time()
        self.report(f"ADMM | mu={mu:.3g} | threshold={threshold:.3e}")

        for iteration in range(1, n_iter + 1):
            if self.interrupted:
                self.report(f"Interrupted at iteration {iteration}.")
                return f

            u = operator.solve_normal(Htg + mu * (f - eta), mu)
            f = torch.max(u + eta - threshold, zero)
            eta = eta + mu * (u - f)

            self.report(f"iter {iteration:4d}/{n_iter}")
            self.publish(f)

        self.report(f"Execution in {time.time() - started:.2f} s.")
        return f

    @staticmethod
    def _threshold(operator, g, like, ratio):
        """A fraction of the measurement scale, divided by the operator's gain (see module doc)."""
        operator_norm = operator.lipschitz(like) ** 0.5
        if operator_norm <= 0:
            return 0.0
        return float(g.max()) * ratio / operator_norm
