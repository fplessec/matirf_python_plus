"""
Utility functions shared across metrics.

Provides:
    - optimal_scale(f, f_true): computes alpha* = argmin ||f_true - alpha * f||^2
    - align_scale(f, f_true): returns (alpha * f, alpha)
    - to_numpy(x): detaches and converts a tensor to numpy
"""

import torch

import common.settings as settings


def optimal_scale(f, f_true, eps=1e-12):
    """Computes alpha* that minimizes ||f_true - alpha * f||^2."""
    f = torch.as_tensor(f, device=settings.device, dtype=torch.float32)
    f_true = torch.as_tensor(f_true, device=settings.device, dtype=torch.float32)
    num = (f * f_true).sum()
    den = (f * f).sum() + eps
    return (num / den).item()


def align_scale(f, f_true):
    """Returns (alpha * f, alpha) where alpha minimizes ||f_true - alpha * f||^2."""
    alpha = optimal_scale(f, f_true)
    return alpha * f, alpha


def to_numpy(x):
    """Detaches and converts a tensor to numpy float32 array."""
    return x.detach().to('cpu').float().numpy()
