"""
Denoiser registry.

Each denoiser is registered with a display name and a callable with a common
interface:  denoiser_fn(image, sigma) -> denoised_image

This registry is the single source of truth for:
    - DENOISER_LIST (used in UI dicts for the option widget)
    - DENOISER_REGISTRY (used at runtime to dispatch by name)

To add a new denoiser, just add an entry to DENOISER_REGISTRY below.
"""

from .denoise_tv_bregman import TVBregmanDenoiser
from .nlridge import NLRidgeDenoiser
from .denoise_gaussian import GaussianDenoiser
from .denoise_bilateral import BilateralDenoiser
from .denoise_wiener import WienerDenoiser
from .denoise_dct import DCTDenoiser


_ALL_DENOISERS = [
    NLRidgeDenoiser(),
    TVBregmanDenoiser(),
    GaussianDenoiser(),
    BilateralDenoiser(),
    WienerDenoiser(),
    DCTDenoiser(),
]

# Maps display name -> callable(image, sigma) -> denoised_image.
# "None" is handled separately (no denoising), so it's not in the registry.
DENOISER_REGISTRY = {d.name: d for d in _ALL_DENOISERS}

# List for the UI option widget (first element = default).
DENOISER_LIST = ["None"] + list(DENOISER_REGISTRY.keys())

# Denoisers that support 3D anisotropy (delta parameter).
ANISOTROPIC_DENOISERS = [d.name for d in _ALL_DENOISERS if d.supports_anisotropy]

# Denoisers that do NOT support 3D natively (will fallback to slice-by-slice).
SLICE_BY_SLICE_DENOISERS = [d.name for d in _ALL_DENOISERS if not d.supports_3d]
