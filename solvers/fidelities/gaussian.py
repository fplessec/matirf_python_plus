import torch.nn.functional as F

from .base import DataFidelity, floored


class GaussianFidelity(DataFidelity):
    """
    Additive Gaussian (read) noise of variance b = sigma^2.

        D(Hf, g) = mean( (Hf - g)^2 ) / (2 b)

    With b = 1 this is v1's 1/2 mean squared error, bit for bit.
    """

    name = "gaussian"
    display_name = "L2 (Gaussian noise)"
    noise_parameters = ("b",)
    formula = r"D(Hf,g) = \frac{1}{2b}\,\overline{(Hf - g)^2}"

    def __init__(self, b: float = 1.0):
        self.b = floored(b)

    def quadratic_scale(self, n_pixels: int) -> float:
        """D = quadratic_scale / 2 * ||Hf - g||^2, the form a least-squares data step needs."""
        return 1.0 / (self.b * n_pixels)

    def loss(self, Hf, g):
        if self.b == 1.0:
            return 0.5 * F.mse_loss(Hf, g)
        return F.mse_loss(Hf, g) / (2.0 * self.b)
