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

    def __init__(self, n_iter=30):
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        components = diff_ops.spatial_grad(f)
        return sum(c.abs() for c in components).mean()

    def prox_sum(self, f, lambda_reg, diff_ops, **kwargs):
        """
        Proximal operator of lambda * sum_x sum_i |D_i f(x)|, the anisotropic TV of `loss`.

        Solved on the dual (Beck & Teboulle, "Fast gradient-based algorithms for constrained
        total variation image denoising and deblurring", IEEE TIP 2009):

            u = f - lambda * div(p),     each p_i(x) in [-1, 1]

        by accelerated projected gradient on p. `div` is minus the adjoint of `spatial_grad`,
        so the step 1 / ||D||^2 — one per axis, delta^2 for the weighted axial one — is
        exactly the one the theory allows.

        CHANGE FROM V1: v1 used Chambolle's semi-implicit iteration with the sign of the dual
        step reversed and no 1/lambda factor, and projected onto the ISOTROPIC ball while
        `loss` is anisotropic. The result was worse than no denoising at all:
        1/2 ||u - f||^2 + lambda TV(u) came out above its value at u = f.
        """
        if lambda_reg <= 0:
            return f.clone()
        ndim = f.dim()
        delta = getattr(diff_ops, "delta", 1.0)
        norm_squared = (ndim - 1) + (delta ** 2 if ndim == 3 else 1.0)
        step = 1.0 / (lambda_reg * norm_squared)

        p = [torch.zeros_like(f) for _ in range(ndim)]
        q = [torch.zeros_like(f) for _ in range(ndim)]     # the extrapolated point
        t_k = 1.0
        for _ in range(self.n_iter):
            u = f - lambda_reg * diff_ops.divergence(*q)
            grads = diff_ops.spatial_grad(u)
            previous = p
            p = [(q_i - step * g_i).clamp_(-1.0, 1.0) for q_i, g_i in zip(q, grads)]
            t_next = (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5) / 2.0
            momentum = (t_k - 1.0) / t_next
            q = [p_i + momentum * (p_i - old_i) for p_i, old_i in zip(p, previous)]
            t_k = t_next

        return f - lambda_reg * diff_ops.divergence(*p)
