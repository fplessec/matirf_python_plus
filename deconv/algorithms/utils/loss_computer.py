import torch
import torch.nn.functional as F

from deconv.core.operations import apply_psf


## 2D forward finite-difference gradient, returns (dx, dy) with same shape as f:
def _grad_xy(f: torch.Tensor):
    # dy = f[i+1, j] - f[i, j]
    dy = torch.zeros_like(f)
    dy[:-1, :] = f[1:, :] - f[:-1, :]
    # dx = f[i, j+1] - f[i, j]
    dx = torch.zeros_like(f)
    dx[:, :-1] = f[:, 1:] - f[:, :-1]
    return dx, dy


## callable loss: L(f) = (1 - lambda_reg) * 1/2 ||Hf - g||^2 + lambda_reg * R(f):
class LossComputer:

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
