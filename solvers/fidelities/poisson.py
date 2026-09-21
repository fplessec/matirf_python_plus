import torch

from .base import DataFidelity, floored


class PoissonFidelity(DataFidelity):
    """
    Photon (Poisson) noise: g = a * Poisson(Hf / a), a = 1 / N.

        D(Hf, g) = mean( Hf - g + g log(g / Hf) ) / a

    The Kullback-Leibler divergence between g and Hf, i.e. the Poisson negative
    log-likelihood of the photon counts g / a, up to a constant that makes D(g, g) = 0.
    Assumes g >= 0 (guaranteed by the preprocessing); Hf is floored at a tiny positive value.
    """

    name = "poisson"
    display_name = "KL divergence (Poisson noise)"
    noise_parameters = ("a",)
    formula = (r"D(Hf,g) = \frac{1}{a\,n_g}\sum_{i=1}^{n_g}"
               r"\left(Hf - g + g\,\log\frac{g}{Hf}\right)_i")

    def __init__(self, a: float = 1.0):
        self.a = floored(a)

    def loss(self, Hf, g):
        Hf_safe = Hf.clamp(min=1e-12)
        kl = Hf_safe - g + torch.xlogy(g, g) - torch.xlogy(g, Hf_safe)
        return kl.mean() / self.a
