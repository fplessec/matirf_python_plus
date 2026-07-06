"""
Normalized Mean Squared Error:  ||alpha*f - f_true||^2 / ||f_true||^2

Always uses scale alignment (the metric is defined that way).
"""

import torch

import common.settings as settings
from common.core.metrics.base import Metric, optimal_scale


class NMSE(Metric):

    name = "NMSE"
    requires = set()

    def compute(self, f, f_true, features=set(), **kw):
        f = torch.as_tensor(f, device=settings.device)
        f_true = torch.as_tensor(f_true, device=settings.device)
        alpha = optimal_scale(f, f_true)
        num = torch.sum((alpha * f - f_true) ** 2)
        den = torch.sum(f_true ** 2) + 1e-12
        return (num / den).item()
