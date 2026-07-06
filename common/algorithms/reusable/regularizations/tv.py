import torch

from .base import Regularization


class TVRegularization(Regularization):
    """
    Total Variation regularization (L1 norm of the gradient):
        R(f) = ||∇f||₁ = sum(|∂f/∂xi|)

    Promotes piecewise-constant solutions (edge-preserving).
    Works in 2D and 3D via diff_ops.
    """

    name = "tv"
    display_name = "L1 norm of the gradient"
    uses_diff_ops = True

    def __init__(self, tau=0.125, n_iter=20):
        self.tau = tau
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        components = diff_ops.spatial_grad(f)
        return sum(c.abs() for c in components).mean()

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        """
        Proximal Operator of the L1 norm of the gradient (Total Variation regularization) using Chambolle's method
        Args:
            f: input image (Y,X) or (Z,Y,X)
            lambda_reg: regularization coefficient
            tau: dual step size (<= 1/6 for stability in 3D) ; default 1/8
            n_iter: number of Chambolle iterations (empirical 20–50)
        """
        ndim = f.dim()
        # pre-allocated dual variables:
        p = [torch.zeros_like(f) for _ in range(ndim)]
        d = [torch.empty_like(f) for _ in range(ndim)]
        u = f.clone()
        div = torch.empty_like(f)

        for _ in range(self.n_iter):
            # primal update
            div[:] = diff_ops.divergence(*p)
            u[:] = f - lambda_reg * div
            # dual update
            grads = diff_ops.spatial_grad(u)
            for i in range(ndim):
                d[i][:] = grads[i]
            denom = 1.0 + self.tau * torch.sqrt(sum(di ** 2 for di in d) + 1e-12)
            for i in range(ndim):
                p[i][:] = (p[i] + self.tau * d[i]) / denom

        div[:] = diff_ops.divergence(*p)
        return f - lambda_reg * div
