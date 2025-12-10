from typing import Dict, Any
import time

import torch

from algorithms.abstract_algo import Algorithm
from operations import apply_matirf_operator, get_variables_from_dict
from .utils import *
from settings import device, dtype


class PytorchOptimization(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):

        (max_iter, lr, K, EPS, reg, coeff, rho,
         forced_pos, gamma, random_init) = get_variables_from_dict(params, ['max_iter', 'lr',
                                    'K', 'EPS', 'reg', 'coeff', 'rho', 'forced_pos', 'gamma', 'random_init'])

        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        #  f0 = ( HtH + gamma Id )^(-1) Htg
        f = apply_matirf_operator(torch.inverse(HtH + gamma * identity), Htg)
        if random_init:
            torch.manual_seed(123)
            f = torch.rand(f.shape[0], f.shape[1], f.shape[2])
        f.requires_grad = True
        optimizer = torch.optim.Adam([f], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        self.LOSS = {"loss": [],
                     "lr": []}
        t0 = time.time()
        loss = compute_loss(f, g, H, reg, coeff, rho)
        self._print(f"iter{0}, loss = {loss.item():.8f}, lr = {get_lr(optimizer)}")
        prev_loss = loss

        for iter in range(max_iter):
            if self.is_stop_requested():
                return None
            optimizer.zero_grad()
            loss = compute_loss(f, g, H, reg, coeff, rho)
            loss.backward()
            optimizer.step()
            if forced_pos:
                with torch.no_grad():
                    f[f < 0] = 0

            self.LOSS["loss"].append(loss.item())
            self.LOSS["lr"].append(get_lr(optimizer))
            if iter % K == K - 1:
                self._print(f"iter{iter + 1}, loss = {loss.item():.8f}, "
                                           f"lr = {get_lr(optimizer)}, dloss = {loss - prev_loss:+.8f}")

            if iter % K == K - 1:
                # prev loss c'est la loss il y a K itérations
                if loss - prev_loss >= 0:
                    scheduler.step()
                elif loss - prev_loss > - EPS:
                    break
                prev_loss = loss

        self._print(f"execution en {time.time() - t0} s")
        return f


