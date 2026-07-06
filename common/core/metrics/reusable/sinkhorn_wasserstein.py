"""
Sinkhorn-regularized Wasserstein distance (optimal transport).

Only meaningful for 3D data: uses (Z, Y, X) coordinates with
anisotropy ratio delta.
"""

# TODO: review (optimal transport)

import torch

import common.settings as settings
from common.core.metrics.base import Metric
from common.core.features import THREE_D


class SinkhornWasserstein(Metric):

    name = "Sinkhorn Wasserstein"
    requires = {THREE_D}

    def compute(self, f, f_true, features=set(),
                delta=1.0, reg=1e-2, max_iter=100, tol=1e-6,
                eps=1e-12, max_points=5000, **kw):
        f = torch.as_tensor(f, device=settings.device).float()
        f_true = torch.as_tensor(f_true, device=settings.device).float()
        Z, Y, X = f_true.shape
        f = f.reshape(-1)
        t = f_true.reshape(-1)
        f_sum = f.sum()
        t_sum = t.sum()
        if f_sum < eps or t_sum < eps:
            return float("nan")
        mu = f / f_sum
        nu = t / t_sum
        zz, yy, xx = torch.meshgrid(
            torch.arange(Z, device=settings.device),
            torch.arange(Y, device=settings.device),
            torch.arange(X, device=settings.device),
            indexing='ij'
        )
        coords = torch.stack([
            zz * delta,
            yy,
            xx
        ], dim=-1).reshape(-1, 3).float()
        idx_f = torch.where(mu > 0)[0]
        idx_t = torch.where(nu > 0)[0]
        if len(idx_f) > max_points:
            idx_f = idx_f[torch.randperm(len(idx_f))[:max_points]]
        if len(idx_t) > max_points:
            idx_t = idx_t[torch.randperm(len(idx_t))[:max_points]]
        coords_f = coords[idx_f]
        coords_t = coords[idx_t]
        mu = mu[idx_f]
        nu = nu[idx_t]
        mu = mu / (mu.sum() + eps)
        nu = nu / (nu.sum() + eps)
        C = torch.cdist(coords_f, coords_t) ** 2
        K = torch.exp(-C / reg)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for _ in range(max_iter):
            u_prev = u
            u = mu / (K @ v + eps)
            v = nu / (K.t() @ u + eps)
            if torch.max(torch.abs(u - u_prev)) < tol:
                break
        gamma = torch.diag(u) @ K @ torch.diag(v)
        W = torch.sum(gamma * C)
        return torch.sqrt(W).item()
