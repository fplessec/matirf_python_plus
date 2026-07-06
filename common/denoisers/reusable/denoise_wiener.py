import torch

import common.settings as settings
from common.denoisers.base import Denoiser
from common.denoisers.base.utils import _from_5d, _to_5d, _is_3d, _conv_reflect_5d


class WienerDenoiser(Denoiser):

    name = "Wiener"
    supports_3d = True
    supports_anisotropy = False

    def denoise(self, y, sigma, delta=1., window_size=None):
        """
        Wiener denoising of a 2D or 3D image with additive Gaussian noise.
        Estimates local mean and variance within a uniform window, and
        applies adaptive smoothing based on the noise variance.

        Each output pixel x̂(i) is computed from the local statistics:
            x̂(i) = μ(i) + g(i) · (y(i) - μ(i))
            where:
                μ(i) = [sum over j of] K(j) · y(i - j)  (local mean)
                var(i) = [sum over j of] K(j) · y(i - j)^2 - μ(i)^2  (local variance)
                g(i) = max(var(i) - sigma^2, 0) / var(i)  (Wiener gain)
            and K is a normalized uniform kernel:
                K(j) = 1 / N   over the local window
                sum_j K(j) = 1

        Args:
            y (Tensor): Input noisy image (1,Y,X) or (Z,Y,X)
            sigma (float): Noise standard deviation
            delta (float): Unused (kept for API consistency)
            window_size (int, optional): Size of the local window
        Returns:
            Tensor: Denoised image with same shape as input
        """
        y5d = _to_5d(y)
        is3d = _is_3d(y)
        if window_size is None:
            window_size = int(max(5, round(2 * sigma)))
            window_size += (window_size % 2 == 0)
            # window size ~ 2*sigma (more noise => larger neighborhood),
            # with a minimum of 5 for stable statistics, and forced odd for a centered window
        if is3d:
            K = torch.ones((window_size,) * 3, device=settings.device, dtype=settings.dtype)
        else:
            K = torch.ones((window_size, window_size), device=settings.device, dtype=settings.dtype)
        # normalize uniform kernel:
        K /= K.numel()
        mu = _conv_reflect_5d(y5d, K)  # μ(i): local mean
        var = _conv_reflect_5d(y5d ** 2, K) - mu ** 2  # var(i): local variance
        gain = torch.clamp(var - sigma**2, min=0.0) / (var + 1e-12)  # Wiener gain g(i)
        x5d = mu + gain * (y5d - mu)  # x̂(i) = μ(i) + g(i) · (y(i) - μ(i))
        return _from_5d(x5d)
