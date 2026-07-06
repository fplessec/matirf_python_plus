"""
Angular distance between f and f_true (in radians).

Scale-invariant by nature: arccos(cos(f, f_true)).
"""

import torch

import common.settings as settings
from common.core.metrics.base import Metric


class AngularDistance(Metric):

    name = "angular distance"
    requires = set()

    def compute(self, f, f_true, features=set(), eps=1e-12, **kw):
        f = torch.as_tensor(f, device=settings.device)
        f_true = torch.as_tensor(f_true, device=settings.device)
        f_flat = f.reshape(-1)
        t_flat = f_true.reshape(-1)
        dot = torch.sum(f_flat * t_flat)
        norm_f = torch.norm(f_flat)
        norm_t = torch.norm(t_flat)
        cos = dot / (norm_f * norm_t + eps)
        cos = torch.clamp(cos, -1, 1)
        return torch.acos(cos).item()
