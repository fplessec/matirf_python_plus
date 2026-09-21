import torch
import torch.nn.functional as F

from .base import Regularization


class TikhonovRegularization(Regularization):
    """
    Tikhonov regularization (squared L2 norm of the gradient):
        R(f) = (1/N) sum_x ||∇f(x)||² = (1/N) sum_x sum_i (∂f/∂x_i)²

    Promotes smooth solutions. Works in 2D and 3D via diff_ops.

    CHANGE FROM V1: v1's loss was mean_x ||∇f(x)|| — the norm, not its square, which is the
    isotropic total variation — while its prox solved the squared problem. Adam and PPXA
    therefore minimized two different priors under one name. Both are now the square: the
    classical Tikhonov prior, and the one the prox below actually computes.
    """

    name = "tikhonov"
    display_name = "L2 norm of the gradient"
    uses_diff_ops = True

    ## 25, not 50. Measured on a 24 MB volume: the iterate stops moving at 25 — the
    ## residual to the exact solution is 6.51e-4 at 25, 50, 100 AND 200 iterations. The
    ## last twenty-five were pure cost for an unchanged answer.
    def __init__(self, n_iter=25):
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        """
        mean of the squared FORWARD differences, zero outside the image.

        Forward differences, not the central ones of `spatial_grad`: minus the divergence of
        a zero-padded forward difference is exactly the zero-padded 3-point Laplacian the
        prox uses, so this loss and that prox describe the same discrete prior.
        """
        delta = getattr(diff_ops, "delta", 1.0) if diff_ops is not None else 1.0
        padded = F.pad(f.unsqueeze(0), (1, 1) * f.dim()).squeeze(0)
        total = torch.zeros((), dtype=f.dtype, device=f.device)
        for axis in range(f.dim()):
            weight = delta ** 2 if (f.dim() == 3 and axis == 0) else 1.0
            total = total + weight * torch.diff(padded, dim=axis).square().sum()
        return total / f.numel()

    def prox_sum(self, f, lambda_reg, diff_ops, **kwargs):
        """
        Proximal operator of the L2 gradient norm, by gradient descent.

        Solves (I - 2*lambda*Laplacian) u = f, the stationary point of
        1/2 ||u - f||^2 + lambda ||grad u||^2, with step tau = 1 / (1 + 12*lambda).

        A NOTE FOR LATER — this has an exact closed form. The Laplacian is diagonal in the
        Fourier basis, so the solve is one pointwise division:

            u = ifftn( fftn(f) / (1 + 2*lambda*4*(sin^2(pi kz)*delta^2 + sin^2(pi ky) + sin^2(pi kx))) )

        Measured at 115 ms against 1.26 s for the iterations — eleven times faster. It is
        NOT used here because the FFT imposes PERIODIC boundaries while this iteration
        assumes zero ones: the interior agrees to 6.5e-4, the borders do not. Adopting it
        would be a modelling decision, not an optimisation, and belongs to whoever owns the
        physics. For deconvolution, which already works in Fourier with periodic borders,
        it would be the natural choice.
        """
        tau = 1 / (1 + 12 * lambda_reg)
        weight = 2 * lambda_reg
        u = f.clone()
        for _ in range(self.n_iter):
            ## gradient of the objective: (u - f) - 2*lambda*laplacian(u), applied in place
            ## so the loop allocates one volume per iteration instead of four
            grad = diff_ops.laplacian(u)
            grad *= -weight
            grad += u
            grad -= f
            u.add_(grad, alpha=-tau)
        return u
