"""
REUSABLE DENOISER LAYER (Prebuilt denoising algorithms)

This module provides fully implemented, ready-to-use image denoisers
for inverse problem applications (matirf, deconv, and future extensions).

These denoisers are built on top of the BASE layer and implement
the Denoiser interface with concrete denoising algorithms.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------

While the BASE layer provides the abstract Denoiser class and shared
utilities, the REUSABLE layer provides complete, domain-independent
denoising implementations that can be used directly without modification.

---------------------------------------------------------------------
What belongs here
---------------------------------------------------------------------

This layer includes prebuilt denoisers such as:

- GaussianDenoiser
    Linear smoothing via Gaussian convolution.
    Supports 3D anisotropy.

- BilateralDenoiser
    Edge-preserving smoothing combining spatial and intensity weighting.
    Supports 3D anisotropy.

- WienerDenoiser
    Adaptive smoothing using local statistics (Wiener filter).
    Supports 3D.

- DCTDenoiser
    Transform-domain denoising via soft-thresholding in DCT space.
    Supports 3D.

- TVBregmanDenoiser
    Total variation denoising via split-Bregman optimization.
    2D only (applied slice-by-slice on 3D data).

- NLRidgeDenoiser
    Non-local ridge regression denoising (patch-based).
    2D only (applied slice-by-slice on 3D data).

---------------------------------------------------------------------
Registry
---------------------------------------------------------------------

This layer also provides the denoiser registry, which is the single
source of truth for:
    - DENOISER_REGISTRY: maps display name -> callable denoiser instance
    - DENOISER_LIST: list of display names for UI option widgets
    - ANISOTROPIC_DENOISERS: denoisers supporting the delta parameter
    - SLICE_BY_SLICE_DENOISERS: 2D-only denoisers (fallback on 3D)

---------------------------------------------------------------------
Design role
---------------------------------------------------------------------

Reusable denoisers:
    - implement the Denoiser interface from base/
    - provide denoise(y, sigma, delta) -> denoised tensor
    - remain independent of specific inverse problem implementations
    - are safe to reuse across matirf, deconv, and future modules

---------------------------------------------------------------------
Key distinction
---------------------------------------------------------------------

- base/:
    abstract denoising primitive (Denoiser) and shared utilities

- reusable/:
    concrete denoising algorithms (Gaussian, Bilateral, Wiener, DCT, ...)
"""

from .denoiser_list import (
    DENOISER_LIST, DENOISER_REGISTRY,
    ANISOTROPIC_DENOISERS, SLICE_BY_SLICE_DENOISERS,
)
from .denoise_gaussian import GaussianDenoiser
from .denoise_bilateral import BilateralDenoiser
from .denoise_wiener import WienerDenoiser
from .denoise_dct import DCTDenoiser
from .denoise_tv_bregman import TVBregmanDenoiser
from .nlridge import NLRidgeDenoiser
