import torch

from .base import DataFidelity


class PoissonGaussianFidelity(DataFidelity):
    """
    Data fidelity for mixed Poisson-Gaussian noise.

        D(Hf, g) = sum( (g - Hf)^2 / (2 * (a * Hf + b)) + 0.5 * log(a * Hf + b) )

    where:
        a : gain parameter (Poisson component)
        b : variance of the Gaussian component

    Default values a=1.0, b=0.0 reduce to pure Poisson.
    """

    name = "poisson-gaussian"
    display_name = "Poisson-Gaussian"
    noise_model = "poisson-gaussian"

    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def loss(self, Hf, g):
        var = (self.a * Hf + self.b).clamp(min=1e-12)
        return ((g - Hf) ** 2 / (2 * var) + 0.5 * torch.log(var)).mean()
