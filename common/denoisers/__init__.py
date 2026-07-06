"""
DENOISERS ARCHITECTURE OVERVIEW (Inverse Problems Framework)

This package implements the full denoising stack used by inverse problem
algorithms that rely on implicit priors (Plug-and-Play, MCMC).

Denoisers are autonomous image processing operators: they take a noisy
image and return a denoised version. They are used *by* algorithms
(PnP, MCMC) but do not depend on them.

The system is organized into two layers, from low-level primitives
to ready-to-use implementations.

=====================================================================
1. base/
=====================================================================

Low-level building blocks for image denoising.

This layer defines:
    - Denoiser (abstract base class with __call__ normalization logic)
    - utils (shared tensor helpers: kernel construction, convolution)

It is responsible for:
    - defining the denoiser contract: denoise(y, sigma, delta)
    - normalizing 2D/3D input formats
    - handling slice-by-slice fallback for 2D-only denoisers on 3D data

This layer is intentionally independent of any specific denoiser or problem.

=====================================================================
2. reusable/
=====================================================================

Prebuilt denoising algorithms at the signal processing level.

This layer contains ready-to-use denoisers that can be plugged into
any algorithm without modification:
    - Gaussian, Bilateral, Wiener, DCT, TV Bregman, Non-Local Ridge

It also provides the denoiser registry (DENOISER_REGISTRY, DENOISER_LIST)
used by algorithm ui_params and at runtime for dispatch by name.

=====================================================================

Design summary:

    base     -> denoising primitive and shared utilities
    reusable -> concrete denoising algorithms and registry

Note: unlike common/algorithms and common/gui, this package has no
specializable/ layer. All denoisers are complete implementations that
do not need to be subclassed by specific inverse problems.

This structure makes it easy to:
    - add new denoisers without modifying algorithms
    - reuse denoisers across all inverse problem modules
    - keep a clean separation between denoising and optimization
"""

from .base import Denoiser
from .reusable import (
    DENOISER_LIST, DENOISER_REGISTRY,
    ANISOTROPIC_DENOISERS, SLICE_BY_SLICE_DENOISERS,
    GaussianDenoiser, BilateralDenoiser, WienerDenoiser,
    DCTDenoiser, TVBregmanDenoiser, NLRidgeDenoiser,
)
