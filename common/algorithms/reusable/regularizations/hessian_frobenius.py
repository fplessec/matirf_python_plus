from .base import Regularization


class HessianFrobeniusRegularization(Regularization):
    """
    Frobenius norm of the Hessian regularization:
        R(f) = ||H(f)||_F

    Promotes smooth second-order variations. Requires hessian operator (3D).
    """

    name = "hessian_frobenius"
    display_name = "frobenius norm of the hessian"
    uses_diff_ops = True

    def __init__(self, step=0.1, n_iter=10):
        self.step = step
        self.n_iter = n_iter

    def loss(self, f, diff_ops):
        hess = diff_ops.hessian(f)
        return hess.sum(dim=(0, 1)).mean()

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        """
        Proximal of Frobenius norm of Hessian using gradient descent.

        step: gradient step size
        n_iter: number of iterations
        """
        u = f.clone()
        for _ in range(self.n_iter):
            hess = diff_ops.hessian(u)
            # prox of the hessian norm (sqrt of the squared norm):
            grad = hess.sum(dim=(0, 1))
            u = u - self.step * lambda_reg * grad
        return u
