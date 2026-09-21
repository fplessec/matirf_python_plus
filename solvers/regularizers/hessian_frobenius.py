from . import _hessian
from .base import Regularization


class HessianFrobeniusRegularization(Regularization):
    """
    Hessian-Frobenius regularization — the standard second-order penalty:

        R(f) = sum_x ||H f(x)||_F
             = sum_x sqrt( fxx^2 + fyy^2 + fzz^2 + 2 fxy^2 + 2 fxz^2 + 2 fyz^2 )

    An L1 norm across pixels of the per-pixel Frobenius norm of the Hessian (the Schatten-2
    Hessian norm of Lefkimmiatis et al., IEEE TIP 2012). It favours piecewise-LINEAR images:
    smooth ramps are free, curvature is penalized, and — unlike total variation — gradients
    are not flattened into staircases. Works in 2D and 3D; delta scales the axial direction.

    `loss` returns the mean over interior pixels, as SPITFIRe does. It is exactly SHV with
    weight 1; the discretisation and the prox live in `_hessian.py`.

    CHANGE FROM V1: v1 computed `hessian(f).sum(dim=(0, 1)).mean()` — the plain sum of the
    nine Hessian entries, signed and unsquared, which is not a norm (it can be negative, and
    it is zero for curved images whose entries cancel). Its prox was a gradient step on that
    sum. Both are replaced by the standard definition above and its true proximal operator.
    """

    name = "hessian_frobenius"
    display_name = "frobenius norm of the hessian"
    uses_diff_ops = True

    def __init__(self, n_iter=20):
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        return _hessian.norm_map(f, _delta(diff_ops), weight=1.0).mean()

    def prox_sum(self, f, lambda_reg, diff_ops, **kwargs):
        """Proximal operator of lambda_reg * sum_x ||H f(x)||_F, by projected dual iterations."""
        return _hessian.prox(f, lambda_reg, _delta(diff_ops), weight=1.0, n_iter=self.n_iter)


def _delta(diff_ops):
    return getattr(diff_ops, "delta", 1.0) if diff_ops is not None else 1.0
