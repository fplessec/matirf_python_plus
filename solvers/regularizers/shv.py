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
        hess = diff_ops.hessian(f)
        hess_term = hess.sum(dim=(0, 1)).mean()
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
        for _ in range(self.n_iter):
            hess = diff_ops.hessian(u)
            grad_h = hess.sum(dim=(0, 1))
            grad_l1 = torch.sign(u)
            grad = self.rho * grad_h + (1 - self.rho) * grad_l1
            u = u - self.step * lambda_reg * grad
        return u
