from .base import Regularization


class L2Regularization(Regularization):
    """
    L2 norm regularization: R(f) = ||f||^2 = sum(f_ijk^2)

    Penalizes large values uniformly.
    """

    name = "l2"
    display_name = "L2 norm"

    def loss(self, f, diff_ops):
        return f.square().mean()

    def prox_sum(self, f, lambda_reg, diff_ops, **kwargs):
        """Proximal of L2 norm: shrinkage."""
        return f / (1 + 2 * lambda_reg)
