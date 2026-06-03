"""
Adam optimizer for 2D deconvolution.

Minimizes the loss defined in algorithms/utils/loss_computer.py:
    L(f) = (1 - lambda_reg) * 1/2 || H f - g ||^2 + lambda_reg * R(f)

Initialization: f0 = g (the blurred image itself).
"""

from typing import Dict, Any
import time

import torch

from base import Algorithm
from ..utils.loss_computer import LossComputer
from deconv.core.operations import get_variables_from_dict


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AdamAlgo(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):
        self.fixe_randomness()

        (max_iter, lr, K, EPS, reg, lambda_reg) = get_variables_from_dict(
            params, ['max_iter', 'lr', 'K', 'EPS', 'reg', 'lambda_reg']
        )

        # initialization: start from g (blurred image), clone and enable grad
        f = g.clone().detach()
        f.requires_grad_(True)

        loss_computer = LossComputer(g=g, H=H, reg=reg, lambda_reg=lambda_reg)
        optimizer = torch.optim.Adam([f], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        t0 = time.time()
        loss = loss_computer(f)
        self._print(
            f"iter 0 \nloss={loss.item():.3e} | "
            f"lr={_get_lr(optimizer):.2e}"
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

            # logging and scheduler
            if it % K == K - 1:
                dloss = loss - prev_loss
                self._print(
                    f"iter {it + 1:4d} \nloss={loss.item():.3e} | "
                    f"lr={_get_lr(optimizer):.2e} | "
                    f"dloss={dloss:+.3e}"
                )
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
