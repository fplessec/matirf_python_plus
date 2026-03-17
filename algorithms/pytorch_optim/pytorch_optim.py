from typing import Dict, Any
import time

import torch

from algorithms.abstract_algo import Algorithm
from operations import apply_matirf_operator, get_variables_from_dict, estimate_delta_anisotropy_from_params
from .utils import get_lr, compute_loss
from settings import device, dtype


class PytorchOptimization(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):

        (max_iter, lr, K, EPS, reg, lambda_reg, delta, rho) = get_variables_from_dict(
          params, ['max_iter', 'lr', 'K', 'EPS', 'reg', 'lambda_reg', 'delta', 'rho'])

        delta_estimated = estimate_delta_anisotropy_from_params(self.measurement_params, self.oper_params)
        self._print(f"delta anisotropy estimation = {delta_estimated}\n")

        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        # f initial: f0 = ( HtH + λ_rr Id )^(-1) Htg   (ridge regression)
        lambda_rr = 10000.  #  lambda_rr >> 1 to stabilize inversion
        f = apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Htg)
        f.requires_grad = True
        optimizer = torch.optim.Adam([f], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        t0 = time.time()
        loss = compute_loss(f, g, H, reg, lambda_reg, rho)
        self._print(
            f"iter 0 \nloss={loss.item():.6e} | "
            f"lr={get_lr(optimizer):.3g} "
        )
        prev_loss = loss

        for iter in range(max_iter):
            if self.is_stop_requested():
                return None

            optimizer.zero_grad()
            loss = compute_loss(f, g, H, reg, lambda_reg, rho, delta=delta)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                f[f < 0] = 0  # orthogonal projection of the negative values on the positive set (forcing positivity)

            if iter % K == K - 1:
                self._print(
                    f"iter {iter + 1:4d} \nloss={loss.item():.6e} | "
                    f"lr={get_lr(optimizer):.3g} | "
                    f"dloss={loss - prev_loss:+.3e}"
                )
                # prev_loss is the loss K iterations before
                if loss - prev_loss >= 0:
                    scheduler.step()  # divides lr by 2
                elif loss - prev_loss > - EPS:  # the stopping criterion is met
                    self._print("\nThe stopping criterion EPS has been met.")
                    self._print(f"Execution en {time.time() - t0} s.")
                    return f
                prev_loss = loss

        self._print("\nThe maximum iterations number has been reached.")
        self._print(f"Execution en {time.time() - t0} s.")
        return f


