import torch
import common.settings as settings


class Denoiser:
    """
    Base class for all denoisers.

    Subclasses must override ``denoise(y, sigma, delta)``.
    ``__call__`` normalises the input to (Z, Y, X) with Z=1 for 2D,
    delegates to ``denoise``, and restores the original shape.

    For denoisers that do not support 3D natively (supports_3d = False),
    the base class applies the denoiser slice-by-slice along axis Z.
    """

    name: str = ""
    supports_3d: bool = True
    supports_anisotropy: bool = True

    def __call__(self, y, sigma, delta=1.0):
        original_shape = y.shape
        y = y.to(device=settings.device, dtype=settings.dtype)

        if y.dim() == 2:
            y = y.unsqueeze(0)

        is_3d = y.shape[0] > 1

        if is_3d and not self.supports_3d:
            # slice-by-slice fallback for 2D-only denoisers
            slices = []
            for z in range(y.shape[0]):
                slices.append(self.denoise(y[z : z + 1], sigma, delta))
            result = torch.cat(slices, dim=0)
        else:
            result = self.denoise(y, sigma, delta)

        return result.reshape(original_shape)

    def denoise(self, y, sigma, delta=1.0):
        raise NotImplementedError
