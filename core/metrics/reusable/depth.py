"""
Depth metrics for MA-TIRF — the questions only a 3D axial reconstruction answers.

    Depth Error (nm)   at the right depth?   mean |z(f) - z(f_true)| over the columns holding
                                             signal, z = the brightest voxel's plane. THE
                                             MA-TIRF question ("how far from the glass is it?").
                                             Needs the plane spacing dz (nm), passed as a kwarg.
    Stack Recovery     super-resolved in z?  of the columns where the truth holds two structures
                                             >= SEPARATION_PLANES apart, the fraction where f
                                             shows two peaks too.

Both read the depth of a peak (argmax / local maxima), so they are scale-invariant by
construction — no scale alignment is needed even when the problem is scale-ambiguous.
"""

import torch

from core.features import Feature
from core.metrics.base import Metric

## a column holds signal above this fraction of the volume's brightest column
SIGNAL = 0.1
## two peaks count as separate structures at this many planes apart (5 x 6 nm = 30 nm)
SEPARATION_PLANES = 5
## a local maximum counts as a peak above this fraction of its column's maximum
PEAK = 0.2


def _signal_columns(truth: torch.Tensor) -> torch.Tensor:
    lateral = truth.max(dim=0).values
    return lateral > SIGNAL * lateral.max()


def _stacked(volume: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """For each selected column: does it hold two peaks >= SEPARATION_PLANES apart?"""
    col = volume[:, columns]
    inner = col[1:-1]
    peaks = (inner > col[:-2]) & (inner >= col[2:]) & (inner > PEAK * col.max(dim=0).values)
    index = torch.arange(1, col.shape[0] - 1, device=col.device)[:, None].expand_as(peaks)
    first = torch.where(peaks, index, col.shape[0]).min(dim=0).values
    last = torch.where(peaks, index, -1).max(dim=0).values
    return (last - first) >= SEPARATION_PLANES


class DepthError(Metric):

    name = "Depth Error (nm)"
    requires = {Feature.THREE_D}

    def compute(self, f, f_true, features=set(), dz=1.0, **kw):
        f = torch.as_tensor(f).detach().double()
        f_true = torch.as_tensor(f_true).detach().double()
        columns = _signal_columns(f_true)
        if not columns.any():
            return float("nan")
        z_f = f.argmax(dim=0)[columns].double()
        z_t = f_true.argmax(dim=0)[columns].double()
        return float((z_f - z_t).abs().mean() * float(dz))


class StackRecovery(Metric):

    name = "Stack Recovery"
    requires = {Feature.THREE_D}

    def compute(self, f, f_true, features=set(), **kw):
        f = torch.as_tensor(f).detach().double()
        f_true = torch.as_tensor(f_true).detach().double()
        columns = _signal_columns(f_true)
        stacked_truth = _stacked(f_true, columns)
        if not stacked_truth.any():
            return float("nan")
        return float(_stacked(f, columns)[stacked_truth].double().mean())
