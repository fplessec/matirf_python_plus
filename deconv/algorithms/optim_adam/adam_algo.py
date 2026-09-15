"""
Adam optimizer for 2D deconvolution.

Minimizes:  L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

Initialization: BaseAdam's default, f0 = H^t g (the back-projection).
"""

from common.algorithms import BaseAdam
from deconv.algorithms._base import DeconvForwardModel


class AdamAlgo(DeconvForwardModel, BaseAdam):
    """Adam for 2D deconvolution. Forward operator is FFT-based convolution."""
