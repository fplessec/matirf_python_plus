"""
Fourier Shell Correlation (curve over radial shells).

Only meaningful for 3D data. Returns the full FSC curve
plus the mean as a scalar summary.
"""

# TODO: review (Fourier shell correlation)

import torch

import common.settings as settings
from common.core.metrics.base import Metric
from common.core.features import THREE_D


class FSC(Metric):

    name = "FSC"
    requires = {THREE_D}
    result_type = "curve"

    def compute(self, f, f_true, features=set(),
                n_shells=50, eps=1e-12, **kw):
        f = torch.as_tensor(f, device=settings.device).float()
        f_true = torch.as_tensor(f_true, device=settings.device).float()
        F1 = torch.fft.fftn(f)
        F2 = torch.fft.fftn(f_true)
        Z, Y, X = f.shape
        zz, yy, xx = torch.meshgrid(
            torch.fft.fftfreq(Z, device=settings.device),
            torch.fft.fftfreq(Y, device=settings.device),
            torch.fft.fftfreq(X, device=settings.device),
            indexing='ij'
        )
        r = torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2).reshape(-1)
        F1 = F1.reshape(-1)
        F2 = F2.reshape(-1)
        r_max = r.max()
        bins = torch.linspace(0, r_max, n_shells + 1, device=settings.device)
        frequencies = []
        fsc_vals = []
        for i in range(n_shells):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if mask.sum() < 10:
                continue
            num = torch.sum(F1[mask] * torch.conj(F2[mask]))
            den = torch.sqrt(
                torch.sum(torch.abs(F1[mask]) ** 2)
                * torch.sum(torch.abs(F2[mask]) ** 2)
            ) + eps
            fsc = torch.real(num / den)
            freq = ((bins[i] + bins[i + 1]) / 2).item()
            frequencies.append(freq)
            fsc_vals.append(fsc.item())
        if len(fsc_vals) == 0:
            return {"summary": float("nan"), "x": [], "y": [],
                    "xlabel": "spatial frequency", "ylabel": "FSC"}
        return {
            "summary": float(sum(fsc_vals) / len(fsc_vals)),
            "x": frequencies,
            "y": fsc_vals,
            "xlabel": "spatial frequency",
            "ylabel": "FSC",
        }
