import common.settings as settings
from common.denoisers.base import Denoiser
from common.denoisers.base.utils import _from_5d, _to_5d, _is_3d, _kernel_gaussian, _conv_reflect_5d


class GaussianDenoiser(Denoiser):

    name = "Gaussian"
    supports_3d = True
    supports_anisotropy = True

    def denoise(self, y, sigma, delta=1.):
        """
        Gaussian denoising of a 2D or 3D image with additive Gaussian noise.
        Applies a linear Gaussian filter (convolution) to smooth the image.
        In 3D, accounts for anisotropy using `delta`.

        Each output pixel x̂(i) is a weighted average of its neighbors:
            x̂(i) = [sum over j of] K(j) · y(i - j)
            where K is a normalized Gaussian kernel:
                K(j) ∝ exp( -||j||^2 / (2 * sigma^2) )
                sum_j K(j) = 1

        Note on `sigma`: it is a *noise level* on the [0, 255] scale (shared
        convention with the other denoisers), not a raw kernel width. A Gaussian
        filter is inherently a blur, so we map the noise level to a BOUNDED
        spatial width `s_xy = clip(sigma / 25, 0.3, 2.5)` pixels. This keeps the
        denoiser monotonic in strength while preventing the kernel (and its
        im2col convolution) from exploding for large sigma.

        Args:
            y (Tensor): Input noisy image (1,Y,X) or (Z,Y,X)
            sigma (float): Noise standard deviation on the [0, 255] scale
            delta (float): Anisotropy ratio (Δz/Δxy) for 3D data
        Returns:
            Tensor: Smoothed image with same shape as input
        """
        y5d = _to_5d(y)
        is3d = _is_3d(y)
        # noise level -> bounded spatial Gaussian width (in pixels):
        s_xy = min(max(sigma / 25.0, 0.3), 2.5)
        s_z = min(s_xy / max(delta, 1e-3), 3.0)
        K = (_kernel_gaussian(s_xy, s_z, True)
             if is3d
             else _kernel_gaussian(s_xy, is3d=False))
        # normalizing gaussian kernel:
        K /= K.sum()
        x5d = _conv_reflect_5d(y5d, K)
        return _from_5d(x5d)
