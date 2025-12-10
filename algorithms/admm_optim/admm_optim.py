from typing import Dict, Any
import time

import torch

from algorithms.abstract_algo import Algorithm
from operations import apply_matirf_operator, get_variables_from_dict
from settings import device, dtype


"""
https://arxiv.org/pdf/1801.00882
https://github.com/zcshinee/Pol-TIRF/tree/master
"""


class AdmmOptimization(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):

        (iter, mu) = get_variables_from_dict(params, ['iter', 'mu'])
        threshold = g.max() / H.max() * 0.1
        #  f = ( HtH + gamma Id )^(-1) Htg
        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        HH_inv = torch.inverse(HtH + mu * identity)
        f = apply_matirf_operator(HH_inv, Htg)
        zero = torch.zeros_like(f)
        eta = zero.clone()
        t0 = time.time()
        for k in range(iter):
            if self.is_stop_requested():
                return None
            self._print(f"admm_iter = {k + 1}")
            u = apply_matirf_operator(HH_inv, Htg + mu * (f - eta))
            #self._print(f"values in u : MIN = {u.min().item()} ; MAX = {u.max().item()}")
            f = torch.max(u + eta - threshold, zero)
            #self._print(f"values in f : MIN = {f.min().item()} ; MAX = {f.max().item()}")
            eta = eta + mu * (u - f)
        self._print(f"execution in {time.time() - t0} s")
        return f