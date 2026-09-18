"""
BASE DENOISER LAYER (Denoising primitives)

This module defines the lowest-level building blocks used to construct
image denoisers for inverse problem applications.

It implements two components:

---------------------------------------------------------------------
1. Denoiser
---------------------------------------------------------------------
Abstract base class for all denoisers.

It is responsible for:
    - normalizing input tensors to (Z, Y, X) format
    - dispatching to the concrete denoise() method
    - applying slice-by-slice fallback for 2D-only denoisers on 3D data
    - restoring the original tensor shape on output

Each concrete denoiser only needs to implement denoise(y, sigma, delta).

This is the atomic denoising unit of the system.

---------------------------------------------------------------------
2. utils
---------------------------------------------------------------------
Low-level tensor manipulation helpers shared by multiple denoisers.

Provides:
    - _to_5d / _from_5d: reshape for conv3d compatibility
    - _is_3d: dimension check
    - _gaussian_1d / _kernel_gaussian: Gaussian kernel construction
    - _conv_reflect_5d: convolution with reflect padding

These utilities are internal to the denoiser package and not part
of the public API.

---------------------------------------------------------------------

Overall design philosophy:
    - Denoiser defines the contract: denoise(y, sigma, delta) -> denoised
    - utils provides shared low-level operations
    - neither depends on any specific denoiser or inverse problem

This module is framework-agnostic with respect to inverse problems:
it only defines generic denoising primitives.
"""

from .denoiser import Denoiser
