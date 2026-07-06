import torch

from .base import Regularization


class NoRegularization(Regularization):
    """No regularization: R(f) = 0."""

    name = "none"
    display_name = "no regularization"

    def loss(self, f, diff_ops):
        return torch.zeros((), dtype=f.dtype, device=f.device)

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        return f
