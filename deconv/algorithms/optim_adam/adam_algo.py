"""
ADAM pour la déconvolution 2D.

On minimise par descente de gradient (Adam) la loss définie dans
algorithms/utils/loss_computer.py :
    L(f) = (1 - λ_reg) * 1/2 || H f - g ||² + λ_reg * R(f)

torch.fft étant nativement compatible autograd, on n'a PAS besoin de calculer
le gradient à la main (apply_psf_adjoint) : `loss.backward()` fait tout.

Initialisation f0 = g (l'image floutée elle-même). Choix simple et stable :
g a déjà la bonne shape, est dans [0, 1] et fournit un bon point de départ.
Une alternative serait f0 = wiener(g, H) mais ça ajoute une dépendance.
"""

from typing import Dict, Any
import time

import torch

from ..abstract_algo import Algorithm
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

        # Initialisation : on part de g (image floutée). Cloner pour ne pas
        # accidentellement modifier g, et activer requires_grad pour Adam.
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

            # Projection sur f >= 0 (les images sont supposées positives)
            with torch.no_grad():
                f.clamp_(min=0.0)

            # Logging + scheduler
            if it % K == K - 1:
                dloss = loss - prev_loss
                self._print(
                    f"iter {it + 1:4d} \nloss={loss.item():.3e} | "
                    f"lr={_get_lr(optimizer):.2e} | "
                    f"dloss={dloss:+.3e}"
                )
                if dloss >= 0:
                    scheduler.step()  # divise lr par 2 (la loss remonte)
                elif dloss > -EPS:
                    self._print("\nThe stopping criterion EPS has been met.")
                    self._print(f"Execution en {time.time() - t0:.2f} s.")
                    return f.detach()
                prev_loss = loss

        self._print("\nThe maximum iterations number has been reached.")
        self._print(f"Execution en {time.time() - t0:.2f} s.")
        return f.detach()
