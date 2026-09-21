import torch

from .base import Regularization


class SHVRegularization(Regularization):
    """
    Sparse Hessian Variation (SHV) regularization:
        R(f) = rho * ||H(f)||_F + (1 - rho) * ||f||₁

    Combines Hessian smoothness with L1 sparsity. Requires hessian operator (3D).
    """

    name = "shv"
    display_name = "sparse hessian variation"
    uses_diff_ops = True

    def __init__(self, rho=0.6, step=0.1, n_iter=10):
        self.rho = rho
        self.step = step
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        ## see HessianFrobeniusRegularization.loss for why hessian_sum rather than hessian
        hess_term = diff_ops.hessian_sum(f).mean()
        l1_term = f.abs().mean()
        return self.rho * hess_term + (1 - self.rho) * l1_term

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        """
        Proximal operator of the Sparse Hessian Variation (SHV).

        rho: weight between Hessian and L1 term
        step: gradient step
        n_iter: number of iterations
        """
        u = f.clone()
        weight = self.step * lambda_reg
        for _ in range(self.n_iter):
            grad = diff_ops.hessian_sum(u)
            grad *= self.rho                                  # in place on our own buffer
            grad.add_(torch.sign(u), alpha=1 - self.rho)
            u -= weight * grad
        return u
