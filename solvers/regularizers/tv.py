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
        p = [torch.zeros_like(f) for _ in range(ndim)]   # the dual variables

        for _ in range(self.n_iter):
            u = f - lambda_reg * diff_ops.divergence(*p)          # primal update
            d = diff_ops.spatial_grad(u)                          # dual update
            denom = self._denominator(d)
            for i in range(ndim):
                p[i].add_(d[i], alpha=self.tau)                   # p += tau * d, in place
                p[i].div_(denom)

        return f - lambda_reg * diff_ops.divergence(*p)

    def _denominator(self, gradients):
        """
        1 + tau * sqrt(sum(d_i^2) + eps) — Chambolle's normalisation.

        Worth writing carefully: measured at 13.8 ms per iteration on a 24 MB volume, it
        cost almost as much as the divergence it accompanies. `sum(d ** 2 for d in ...)`
        allocates a fresh volume per component and several more along the sqrt chain;
        accumulating with `addcmul` and finishing in place gives the identical result in
        5.5 ms. No autograd guard is needed: a proximal operator is never differentiated
        through — only `loss` is.
        """
        total = gradients[0] * gradients[0]
        for component in gradients[1:]:
            total = torch.addcmul(total, component, component)
        total += 1e-12
        total.sqrt_()
        total *= self.tau
        total += 1.0
        return total
