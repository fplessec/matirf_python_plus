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

        Args:
            y (Tensor): Input noisy image (1,Y,X) or (Z,Y,X)
            sigma (float): Standard deviation of the Gaussian kernel
            delta (float): Anisotropy ratio (Δz/Δxy) for 3D data
        Returns:
            Tensor: Smoothed image with same shape as input
        """
        y5d = _to_5d(y)
        is3d = _is_3d(y)
        K = (_kernel_gaussian(sigma, sigma / delta, True)
             if is3d
             else _kernel_gaussian(sigma, is3d=False))
        # normalizing gaussian kernel:
        K /= K.sum()
        x5d = _conv_reflect_5d(y5d, K)
        return _from_5d(x5d)
