"""
Peak Signal-to-Noise Ratio (dB).

Adapts to:
    - scale_ambiguous: aligns scale before computing
    - 2d/3d: uses scikit-image for 2D, manual computation for 3D
"""

import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

import common.settings as settings
from common.core.metrics.base import Metric, to_numpy
from common.core.features import THREE_D


class PSNR(Metric):

    name = "PSNR"
    requires = set()

    def compute(self, f, f_true, features=set(), **kw):
        f = torch.as_tensor(f, device=settings.device)
        f_true = torch.as_tensor(f_true, device=settings.device)
        f, f_true = self._prepare(f, f_true, features)
        if THREE_D in features:
            return self._psnr_3d(f, f_true)
        return self._psnr_2d(f, f_true)

    @staticmethod
    def _psnr_2d(f, f_true):
        """PSNR via scikit-image (standard 2D)."""
        data_range = f_true.max().item() - f_true.min().item()
        return float(skimage_psnr(to_numpy(f_true), to_numpy(f), data_range=data_range))

    @staticmethod
    def _psnr_3d(f, f_true):
        """PSNR computed globally on the volume."""
        mse = torch.mean((f - f_true) ** 2)
        max_val = torch.max(f_true)
        return (10 * torch.log10((max_val ** 2) / (mse + 1e-12))).item()
