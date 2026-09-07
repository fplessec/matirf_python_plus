"""
Generic ADMM (Alternating Direction Method of Multipliers) for inverse problems.

Solves:  min_f  1/2 || Hf - g ||^2  +  soft threshold on f  (positivity + sparsity)

Using the splitting u = f with augmented Lagrangian:
    u-step:  u = (H^t H + mu I)^{-1} (H^t g + mu (f - eta))
    f-step:  f = max(u + eta - threshold, 0)
    eta-step: eta = eta + mu (u - f)

Subclasses must override:
    apply_forward(H, f)                     : computes Hf
    apply_adjoint(H, x)                     : computes H^t x
    solve_linear_system(Htg, v, mu, params) : solves (H^t H + mu I)^{-1} (Htg + mu v)
    init_f(g, H, params)                    : initialization of f
    compute_threshold(g, H, params)         : threshold for the soft-thresholding step
"""

from typing import Dict, Any
import time

import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.admm.ui_params import ADMM_UI_PARAMETERS
from common.utils import get_variables_from_dict


class BaseAdmm(Algorithm):

    name = "ADMM"
    ui_params = ADMM_UI_PARAMETERS
    estimator_type = "MAP"
    uses_denoiser = False
    uses_regularization = False

    def run(self, g, H, params: Dict[str, Any]):
        self.fix_randomness()

        (max_iter, mu) = get_variables_from_dict(params, ['iter', 'mu'])

        self.precompute(g, H, params)

        threshold = self.compute_threshold(g, H, params)
        f = self.init_f(g, H, params)
        Htg = self.apply_adjoint(H, g)
        zero = torch.zeros_like(f)
        eta = zero.clone()

        t0 = time.time()

        for k in range(max_iter):
            if self.is_stop_requested():
                return None
            self._print(f"admm_iter = {k + 1}")
            self._update_figure(f)

            u = self.solve_linear_system(Htg, f - eta, mu, params)
            f = torch.max(u + eta - threshold, zero)
            eta = eta + mu * (u - f)

        self._print(f"execution in {time.time() - t0:.2f} s")
        return f

    def precompute(self, g, H, params):
        """Precomputes operators needed by solve_linear_system. Store as self attributes."""
        pass

    def solve_linear_system(self, Htg, v, mu, params):
        """Solves (H^t H + mu I)^{-1} (Htg + mu v). Must be overridden by subclasses."""
        raise NotImplementedError

    def init_f(self, g, H, params):
        """Initialization of f. Must be overridden by subclasses."""
        raise NotImplementedError

    def compute_threshold(self, g, H, params):
        """Computes the soft-thresholding value. Must be overridden by subclasses."""
        raise NotImplementedError
