"""
Calcul de la loss pour la déconvolution 2D.

Formule :
    L(f) = (1 - λ_reg) * 1/2 || H f - g ||² + λ_reg * R(f)

R(f) selon le type de régularisation :
    - "no regularization"        : R(f) = 0
    - "L2 norm"                  : R(f) = ||f||²
    - "L1 norm"                  : R(f) = ||f||₁
    - "L2 norm of the gradient"  : R(f) = ||∇f||₂  (Tikhonov)
    - "L1 norm of the gradient"  : R(f) = ||∇f||₁  (Total Variation)

Le gradient ∇f est calculé par différences finies sur l'axe x (colonnes) et y
(lignes). On ne calcule pas de Hessien ni de SHV (réservés à la 3D MA-TIRF).
"""

import torch
import torch.nn.functional as F

from deconv.core.operations import apply_psf


def _grad_xy(f: torch.Tensor):
    """
    Gradient 2D par différences finies forward, avec padding "edge"
    (replication des bords) pour conserver la même shape que f.

    Retourne (dx, dy) chacun de shape (M, N).
    """
    # dy = f[i+1, j] - f[i, j]
    dy = torch.zeros_like(f)
    dy[:-1, :] = f[1:, :] - f[:-1, :]
    # dx = f[i, j+1] - f[i, j]
    dx = torch.zeros_like(f)
    dx[:, :-1] = f[:, 1:] - f[:, :-1]
    return dx, dy


class LossComputer:
    """
    Construit une loss callable : loss = loss_computer(f).
    Compatible autograd grâce à apply_psf (basé sur torch.fft).
    """

    def __init__(self, g: torch.Tensor, H: torch.Tensor, reg: str, lambda_reg: float):
        self.g = g
        self.H = H
        self.reg = reg
        self.lambda_reg = lambda_reg

    def _grad_l2_norm(self, f, eps=1e-8):
        dx, dy = _grad_xy(f)
        return torch.sqrt(dx ** 2 + dy ** 2 + eps ** 2)

    def _grad_l1_norm(self, f):
        dx, dy = _grad_xy(f)
        return dx.abs() + dy.abs()

    def _reg_term(self, f: torch.Tensor) -> torch.Tensor:
        if self.reg == "no regularization":
            return torch.zeros((), dtype=f.dtype, device=f.device)
        if self.reg == "L2 norm":
            return f.square().mean()
        if self.reg == "L1 norm":
            return f.abs().mean()
        if self.reg == "L2 norm of the gradient":
            return self._grad_l2_norm(f).mean()
        if self.reg == "L1 norm of the gradient":
            return self._grad_l1_norm(f).mean()
        raise ValueError(f"Unknown regularization {self.reg!r}")

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        Hf = apply_psf(self.H, f)
        data_term = 0.5 * F.mse_loss(Hf, self.g, reduction='mean')

        if self.reg == "no regularization":
            return data_term

        reg_term = self._reg_term(f)
        return (1 - self.lambda_reg) * data_term + self.lambda_reg * reg_term
