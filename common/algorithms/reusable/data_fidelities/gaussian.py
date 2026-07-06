import torch.nn.functional as F

from .base import DataFidelity


class GaussianFidelity(DataFidelity):
    """
    L2 data fidelity for additive Gaussian noise.

        D(Hf, g) = 1/2 ||Hf - g||²
    """

    name = "gaussian"
    display_name = "L2 (Gaussian noise)"
    noise_model = "gaussian"

    def loss(self, Hf, g):
        return 0.5 * F.mse_loss(Hf, g)
