"""
Generic Adam optimizer for inverse problems.

Minimizes:  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

Uses torch.optim.Adam with a StepLR scheduler that halves the learning rate
when the loss stops decreasing, and stops early when |dloss| < EPS.

Subclasses must override:
    apply_forward(H, f)   : computes Hf
    apply_adjoint(H, x)   : computes H^T x

Optionally override:
    init_f(g, H, params)  : initialization of f (default: g.clone())
"""

from typing import Dict, Any
import time

import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.adam.ui_params import ADAM_UI_PARAMETERS
from common.utils import get_variables_from_dict


class BaseAdam(Algorithm):

    name = "ADAM"
    ui_params = ADAM_UI_PARAMETERS
    estimator_type = "MAP"
    uses_denoiser = False
    uses_regularization = True

    def run(self, g, H, params: Dict[str, Any]):
        self.fix_randomness()

        (max_iter, lr, K, EPS) = get_variables_from_dict(
            params, ['max_iter', 'lr', 'K', 'EPS'])

        f = self.init_f(g, H, params)
        f.requires_grad_(True)

        loss_computer = self._create_loss_computer(g, H, params)
        optimizer = torch.optim.Adam([f], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        t0 = time.time()
        loss = loss_computer(f)
        self._print(
            f"iter 0 \nloss={loss.item():.3e} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        prev_loss = loss

        for it in range(max_iter):
            if self.is_stop_requested():
                return None

            optimizer.zero_grad()
            loss = loss_computer(f)
            loss.backward()
            optimizer.step()

            # project onto f >= 0 (images are assumed positive)
            with torch.no_grad():
                f.clamp_(min=0.0)

            # logging, live preview, and scheduler
            if it % K == K - 1:
                dloss = loss - prev_loss
                self._print(
                    f"iter {it + 1:4d} \nloss={loss.item():.3e} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"dloss={dloss:+.3e}"
                )
                self._update_figure(f)
                if dloss >= 0:
                    scheduler.step()  # halve lr (loss is increasing)
                elif dloss > -EPS:
                    self._print("\nThe stopping criterion EPS has been met.")
                    self._print(f"Execution in {time.time() - t0:.2f} s.")
                    return f.detach()
                prev_loss = loss

        self._print("\nThe maximum iterations number has been reached.")
        self._print(f"Execution in {time.time() - t0:.2f} s.")
        return f.detach()

    def init_f(self, g, H, params):
        """Initialization of f. Default: f0 = g. Override for custom init (e.g. ridge regression)."""
        return g.clone().detach()
