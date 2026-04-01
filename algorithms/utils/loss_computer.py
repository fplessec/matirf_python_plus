import torch
import torch.nn.functional as F

from algorithms.utils.differential_operators import DifferentialOperators
from core.operations import apply_matirf_operator


class LossComputer:
    """
    Computes the loss for 3D reconstruction problems with optional regularizations.

    Loss formula:
        L(f) = (1 - λ_reg) * 1/2 || H f - g ||² + λ_reg * R(f)

    where R(f) is the regularization term:
        - none          : R(f) = 0
        - l2            : R(f) = ||f||² = sum(f_ijk²)
        - l1            : R(f) = ||f||₁ = sum(|f_ijk|)
        - grad_l2       : R(f) = ||∇f||² = sum((∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)²)
        - grad_l1       : R(f) = ||∇f||₁ = sum(|∂f/∂x| + |∂f/∂y| + |∂f/∂z|)
        - hessian_frob  : R(f) = ||H(f)||_F² = sum(H_ij²) for Hessian matrix
        - shv           : R(f) = ρ * ||H(f)||_F² + (1-ρ) * ||f||₁
    """
    def __init__(self, f, g, H, reg, lambda_reg, rho, delta=1.):
        self.H = H
        self.g = g
        self.diff_ops = DifferentialOperators(delta=delta)
        self.delta = delta
        self.reg = reg
        self.lambda_reg = lambda_reg
        self.rho = rho

    def grad_l2_sq(self, f):
        dx, dy, dz = self.diff_ops.spatial_grad(f)
        eps = 1e-8
        return torch.sqrt(dx**2 + dy**2 + dz**2 + eps**2)

    def grad_l1(self, f):
        dx, dy, dz = self.diff_ops.spatial_grad(f)
        return dx.abs() + dy.abs() + dz.abs()

    def frobenius_norm(self, matrix, eps=1e-8):
        """Calcul de la norme de Frobenius"""
        return torch.sqrt((matrix ** 2).sum() + eps ** 2)

    def hessian_frob_sq(self, f):
        hess = self.diff_ops.hessian(f)
        return self.frobenius_norm(hess)
        return torch.sqrt((hess ** 2).sum() + eps ** 2)

    def shv(self, f):
        return self.rho * self.hessian_frob_sq(f) + (1 - self.rho) * f.abs()


    def __call__(self, f):
        Hf = apply_matirf_operator(self.H, f)
        data_term = 0.5 * F.mse_loss(Hf, self.g, reduction='mean')
        REGULARIZATION_LIST = ["no regularization",
                               "L2 norm",
                               "L1 norm",
                               "L2 norm of the gradient"
                               "L1 norm of the gradient",
                               "frobenius norm of the hessian",
                               "sparse hessian variation"]

        if self.reg == "no regularization":
            return data_term
        elif self.reg == "L2 norm":
            reg = f.square()
        elif self.reg == "L1 norm":
            reg = f.abs()
        elif self.reg == "L2 norm of the gradient":
            reg = self.grad_l2_sq(f)
        elif self.reg == "L1 norm of the gradient":
            reg = self.grad_l1(f)
        elif self.reg == "frobenius norm of the hessian":
            reg = self.hessian_frob_sq(f)
        elif self.reg == "sparse hessian variation":
            reg = self.shv(f)

        elif self.reg == "L1":
            reg = f.abs()
        elif self.reg == "Tikhonov":
            reg = self.grad_l2_sq(f)
        elif self.reg == "Tikhonov Boulanger":
            reg = self.grad_l2_sq(f)
        elif self.reg == "TV":
            reg = self.grad_l1(f)
        elif self.reg == "TV normalized":
            reg = self.grad_l1(f)
        elif self.reg == "Hessian Frobenius":
            reg = self.hessian_frob_sq(f)
        elif self.reg == "SHV":
            reg = self.shv(f)





        else:
            raise ValueError(f"Unknown reg_type {self.reg}")

        return (1 - self.lambda_reg) * data_term + self.lambda_reg * reg.mean()