"""
Pearson correlation coefficient between f and f_true.

Scale and offset invariant by nature.
"""

import torch

import common.settings as settings
from common.core.metrics.base import Metric


class Correlation(Metric):

    name = "Correlation"
    requires = set()

    def compute(self, f, f_true, features=set(), eps=1e-12, **kw):
        f = torch.as_tensor(f, device=settings.device)
        f_true = torch.as_tensor(f_true, device=settings.device)
        f_flat = f.reshape(-1)
        t_flat = f_true.reshape(-1)
        f_mean = torch.mean(f_flat)
        t_mean = torch.mean(t_flat)
        num = torch.sum((f_flat - f_mean) * (t_flat - t_mean))
        den = (torch.sqrt(torch.sum((f_flat - f_mean) ** 2))
               * torch.sqrt(torch.sum((t_flat - t_mean) ** 2)) + eps)
        return (num / den).item()
