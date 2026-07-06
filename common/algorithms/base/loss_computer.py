"""
Loss computer for inverse problems.

Assembles a data fidelity term and a regularization term:
    L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)
"""


class LossComputer:
    """Pluggable loss: data fidelity D + regularization R weighted by lambda_reg."""

    def __init__(self, g, H, apply_forward, data_fidelity, regularization, diff_ops,
                 lambda_reg=0.):
        self.g = g
        self.H = H
        self.apply_forward = apply_forward
        self.data_fidelity = data_fidelity
        self.regularization = regularization
        self.diff_ops = diff_ops
        self.lambda_reg = lambda_reg

    def __call__(self, f):
        Hf = self.apply_forward(self.H, f)
        data_term = self.data_fidelity.loss(Hf, self.g)

        if self.lambda_reg == 0.:
            return data_term

        reg_term = self.regularization.loss(f, self.diff_ops)
        return (1 - self.lambda_reg) * data_term + self.lambda_reg * reg_term
