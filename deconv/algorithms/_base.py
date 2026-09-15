"""
Shared deconvolution specialization bits.

    DeconvForwardModel   the forward model (Hf / H^t g via FFT convolution) — identical for
                         every deconvolution algorithm, so it lives here once (mixin).
"""

from deconv.core.operations import apply_psf


class DeconvForwardModel:
    """The 2D-deconvolution forward model: shared by every deconv algorithm (mixin, no state)."""

    supported_features = {"2d"}

    def apply_forward(self, H, f):
        return apply_psf(H, f)

    def apply_adjoint(self, H, x):
        return apply_psf(H, x, adjoint=True)
