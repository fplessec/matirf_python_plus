class Regularization:
    """
    Base class for regularization terms R(f).

    ----------
    > The scale convention :
    ----------

    R is a MEAN over the voxels of f, like the data fidelity is a mean over the pixels of g
    (solvers/fidelities/base.py). Both terms of L = (1 - lambda) D + lambda R are then
    independent of the image size, and lambda_reg keeps its meaning from one problem to the
    next.

        loss(f)                R(f) = (1 / N_f) sum_x r(f)(x)
        prox_sum(f, w)         argmin_u 1/2 ||u - f||^2 + w * sum_x r(u)(x)
        prox(f, w)             argmin_u 1/2 ||u - f||^2 + w * R(u)  =  prox_sum(f, w / N_f)

    A subclass writes `loss` and `prox_sum` — the textbook proximal operators are stated for
    the sum, so that is the form they are written in — and inherits `prox`, the one solvers
    call. Writing the two on different scales is the mistake this split prevents: a proximal
    solver and a gradient solver would otherwise minimize different objectives.

    diff_ops is a DifferentialOperators instance that provides spatial_grad,
    divergence, laplacian, hessian — works in 2D or 3D transparently.
    """

    name = ""
    display_name = ""
    requires = set()
    uses_diff_ops = False

    def loss(self, f, diff_ops):
        """R(f) -> scalar, a mean over the voxels."""
        raise NotImplementedError

    def prox_sum(self, f, weight, diff_ops, **kwargs):
        """Proximal operator of weight * sum_x r(f)(x) (see the class docstring)."""
        raise NotImplementedError

    def prox(self, f, weight, diff_ops, **kwargs):
        """Proximal operator of weight * R(f), R being the mean `loss` returns."""
        return self.prox_sum(f, weight / f.numel(), diff_ops, **kwargs)
