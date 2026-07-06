import torch

from .base import Regularization


class TikhonovRegularization(Regularization):
    """
    Tikhonov regularization (L2 norm of the gradient):
        R(f) = ||∇f||₂ = sqrt(sum((∂f/∂xi)²))

    Promotes smooth solutions. Works in 2D and 3D via diff_ops.
    """

    name = "tikhonov"
    display_name = "L2 norm of the gradient"
    uses_diff_ops = True

    def __init__(self, n_iter=50):
        self.n_iter = n_iter

    def loss(self, f, diff_ops, eps=1e-8):
        components = diff_ops.spatial_grad(f)
        return torch.sqrt(sum(c ** 2 for c in components) + eps ** 2).mean()

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        """
        Proximal operator of L2 gradient norm (Tikhonov).
        Args:
            f: input image (Y,X) or (Z,Y,X)
            lambda_reg: regularization coefficient
            tau: step size for gradient descent ; default = 1 / (1 + 12*lambda_reg)
            n_iter: number of iterations
        """
        tau = 1 / (1 + 12 * lambda_reg)
        u = f.clone()
        for _ in range(self.n_iter):
            lap = diff_ops.laplacian(u)
            grad = u - f - 2 * lambda_reg * lap
            u = u - tau * grad
        return u
