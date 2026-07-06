class Regularization:
    """
    Base class for regularization terms R(f).

    Each subclass provides:
        - loss(f, diff_ops)                    : R(f) value (for gradient-based algorithms like Adam)
        - prox(f, lambda_reg, diff_ops, **kw)  : proximal operator (for proximal algorithms like PPXA)

    diff_ops is a DifferentialOperators instance that provides spatial_grad,
    divergence, laplacian, hessian — works in 2D or 3D transparently.
    """

    name = ""
    display_name = ""
    requires = set()
    uses_diff_ops = False

    def loss(self, f, diff_ops):
        """R(f) → scalar."""
        raise NotImplementedError

    def prox(self, f, lambda_reg, diff_ops, **kwargs):
        """Proximal operator of lambda_reg * R(f)."""
        raise NotImplementedError
