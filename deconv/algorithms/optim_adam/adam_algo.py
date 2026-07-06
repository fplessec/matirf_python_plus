"""
Adam optimizer for 2D deconvolution.

Minimizes:  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

Initialization: f0 = g (the blurred image itself).
"""

from common.algorithms import BaseAdam
from deconv.core.operations import apply_psf


class AdamAlgo(BaseAdam):
    """Adam for 2D deconvolution. Forward operator is FFT-based convolution."""

    supported_features = {"2d"}

    def apply_forward(self, H, f):
        return apply_psf(H, f)

    def apply_adjoint(self, H, x):
        return apply_psf(H, x, adjoint=True)
