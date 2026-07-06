"""
Structural Similarity Index (SSIM).

Adapts to:
    - scale_ambiguous: aligns scale before computing
    - 3d: computes slice-by-slice along Z axis and averages
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim

import common.settings as settings
from common.core.metrics.base import Metric, to_numpy
from common.core.features import THREE_D


class SSIM(Metric):

    name = "SSIM"
    requires = set()

    def compute(self, f, f_true, features=set(), **kw):
        f = torch.as_tensor(f, device=settings.device)
        f_true = torch.as_tensor(f_true, device=settings.device)
        f, f_true = self._prepare(f, f_true, features)
        f_np = to_numpy(f)
        t_np = to_numpy(f_true)
        data_range = t_np.max() - t_np.min()
        if THREE_D in features:
            return self._ssim_3d(f_np, t_np, data_range)
        return self._ssim_2d(f_np, t_np, data_range)

    @staticmethod
    def _ssim_2d(f_np, t_np, data_range):
        """SSIM via scikit-image (standard 2D)."""
        return float(skimage_ssim(t_np, f_np, data_range=data_range))

    @staticmethod
    def _ssim_3d(f_np, t_np, data_range):
        """SSIM computed slice-by-slice along Z and averaged."""
        scores = []
        for z in range(f_np.shape[0]):
            s = skimage_ssim(t_np[z], f_np[z], data_range=data_range)
            scores.append(s)
        return float(np.mean(scores))
