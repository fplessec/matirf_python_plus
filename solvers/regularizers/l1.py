import torch

from .base import Regularization


class L1Regularization(Regularization):
    """
    L1 norm regularization: R(f) = ||f||₁ = sum(|f_ijk|)

    Promotes sparsity in the image domain.
    """

    name = "l1"
    display_name = "L1 norm"

    def loss(self, f, diff_ops):
        return f.abs().mean()

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        """Soft-thresholding proximal for L1 norm"""
        return torch.sign(f) * torch.clamp(torch.abs(f) - lambda_reg, min=0)
