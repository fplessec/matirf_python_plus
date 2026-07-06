import torch

from .base import DataFidelity


class PoissonFidelity(DataFidelity):
    """
    Kullback-Leibler data fidelity for Poisson noise.

        D(Hf, g) = sum( Hf - g * log(Hf) )

    Assumes g >= 0 and Hf > 0.
    """

    name = "poisson"
    display_name = "KL divergence (Poisson noise)"
    noise_model = "poisson"

    def loss(self, Hf, g):
        Hf_safe = Hf.clamp(min=1e-12)
        return (Hf_safe - g * torch.log(Hf_safe)).mean()
