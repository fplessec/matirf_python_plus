"""
Mean Absolute Error.

Adapts to scale-ambiguous problems by aligning scale first.
"""

import torch

import common.settings as settings
from common.core.metrics.base import Metric


class MAE(Metric):

    name = "MAE"
    requires = set()

    def compute(self, f, f_true, features=set(), **kw):
        f = torch.as_tensor(f, device=settings.device)
        f_true = torch.as_tensor(f_true, device=settings.device)
        f, f_true = self._prepare(f, f_true, features)
        return torch.mean(torch.abs(f - f_true)).item()
