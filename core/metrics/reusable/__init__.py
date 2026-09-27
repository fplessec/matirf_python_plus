"""
REUSABLE METRICS LAYER (Prebuilt quality metrics)

This module provides fully implemented, ready-to-use quality metrics
for evaluating inverse problem reconstructions.

These metrics are built on top of the BASE layer and implement
the Metric interface with concrete computation logic.

Each metric adapts its behavior based on the problem's features:
    > "scale_ambiguous": aligns scale before comparing
    > "3d": handles volumetric data (slice-by-slice SSIM, etc.)
    > "2d": uses standard 2D implementations

The requires set filters metrics that only make sense in certain
contexts (e.g. FSC requires 3D data).

---------------------------------------------------------------------
Registry
---------------------------------------------------------------------

METRIC_REGISTRY: maps display name -> Metric instance
METRIC_LIST: ordered list of display names
"""

from .angular_distance import AngularDistance
from .correlation import Correlation
from .cosine_similarity import CosineSimilarity
from .depth import DepthError, StackRecovery
from .fsc import FSC
from .mae import MAE
from .mse import MSE
from .nmse import NMSE
from .psnr import PSNR
from .scale_alpha import ScaleAlpha
from .sinkhorn_wasserstein import SinkhornWasserstein
from .ssim import SSIM

_ALL_METRICS = [
    ScaleAlpha(),
    MSE(),
    MAE(),
    NMSE(),
    AngularDistance(),
    CosineSimilarity(),
    Correlation(),
    PSNR(),
    SSIM(),
    SinkhornWasserstein(),
    FSC(),
    DepthError(),      # 3D: axial localization (MA-TIRF)
    StackRecovery(),   # 3D: axial super-resolution (MA-TIRF)
]

METRIC_REGISTRY = {m.name: m for m in _ALL_METRICS}
METRIC_LIST = [m.name for m in _ALL_METRICS]
