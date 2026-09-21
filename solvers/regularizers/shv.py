from . import _hessian
from .base import Regularization
from .hessian_frobenius import _delta


class SHVRegularization(Regularization):
    """
    Sparse Hessian Variation — the regularization of SPITFIRe (S. Prigent et al.):

        R(f) = sum_x sqrt( rho^2 ||H f(x)||_F^2 + (1 - rho)^2 f(x)^2 )

    One square root around BOTH terms, which is what distinguishes it from a sum of a
    Hessian penalty and an L1 penalty: a pixel pays for curvature and for brightness
    jointly, so the result is sparse (most pixels exactly dark) AND smooth where it is not.
    rho in [0, 1] trades the two — 1 is pure Hessian-Frobenius, 0 pure L1 sparsity. The
    SPITFIRe default is 0.6.

    The stencils are SPITFIRe's `hv_loss` / `hv_loss_3d`, reproduced exactly (centred second
    derivatives, forward mixed derivatives, delta^2 on fzz, delta on fxz and fyz, interior
    pixels only); `solvers/regularizers/_tests.py` checks it against a verbatim copy.

    CHANGE FROM V1: v1 computed rho * (sum of the nine Hessian entries) + (1 - rho) * |f|,
    i.e. neither the Frobenius norm nor the joint square root. Its prox mixed a gradient
    step with a sign step. Both are replaced by the SPITFIRe definition and its true
    proximal operator (see `_hessian.py`).
    """

    name = "shv"
    display_name = "sparse hessian variation"
    uses_diff_ops = True

    def __init__(self, rho=0.6, n_iter=20):
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [0, 1], got {rho}")
        self.rho = rho
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        return _hessian.norm_map(f, _delta(diff_ops), weight=self.rho).mean()

    def prox_sum(self, f, lambda_reg, diff_ops, **kwargs):
        """Proximal operator of lambda_reg * R(f), by projected dual iterations."""
        return _hessian.prox(f, lambda_reg, _delta(diff_ops), weight=self.rho,
                             n_iter=self.n_iter)
