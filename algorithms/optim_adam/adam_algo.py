from typing import Dict, Any
import time

import torch

from algorithms.abstract_algo import Algorithm
from operations import apply_matirf_operator, get_variables_from_dict, estimate_delta_anisotropy_from_params
from settings import device, dtype
from ..utils import LossComputer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AdamAlgo(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):

        (max_iter, lr, K, EPS, reg, lambda_reg, delta, rho) = get_variables_from_dict(
            params, ['max_iter', 'lr', 'K', 'EPS', 'reg', 'lambda_reg', 'delta', 'rho'])

        delta_estimated = estimate_delta_anisotropy_from_params(self.measurement_params, self.oper_params)
        self._print(f"delta anisotropy estimation = {delta_estimated}\n")

        # Pré-calculs matriciels
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        # f initial: f0 = ( HtH + λ_rr Id )^(-1) Htg   (ridge regression)
        lambda_rr = 10000.  #  lambda_rr >> 1 to stabilize inversion
        f = apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Htg)
        f.requires_grad = True

        loss_computer = LossComputer(f, g, H, reg=reg, lambda_reg=lambda_reg, rho=rho, delta=delta)
        optimizer = torch.optim.Adam([f], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        t0 = time.time()
        loss = loss_computer(f)
        self._print(
            f"iter 0 \nloss={loss.item():.3e} | "
            f"lr={get_lr(optimizer):.2e}"
        )
        prev_loss = loss

        for iter in range(max_iter):
            if self.is_stop_requested():
                return None

            optimizer.zero_grad()
            loss = loss_computer(f)
            loss.backward()
            optimizer.step()

            # Projection sur f >= 0
            with torch.no_grad():
                f.clamp_(min=0)

            # Logging et scheduler
            if iter % K == K - 1:
                dloss = loss - prev_loss
                self._print(
                    f"iter {iter + 1:4d} \nloss={loss.item():.3e} | "
                    f"lr={get_lr(optimizer):.2e} | "
                    f"dloss={dloss:+.3e}"
                )
                if dloss >= 0:
                    scheduler.step()  # divise lr par 2
                elif dloss > -EPS:
                    self._print("\nThe stopping criterion EPS has been met.")
                    self._print(f"Execution en {time.time() - t0:.2f} s.")
                    return f
                prev_loss = loss

        self._print("\nThe maximum iterations number has been reached.")
        self._print(f"Execution en {time.time() - t0:.2f} s.")
        return f