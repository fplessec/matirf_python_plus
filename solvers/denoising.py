"""
Resolving a denoiser by name, and the intensity-scale bridge it needs.

The scale bridge, which is easy to get wrong
--------------------------------------------
A reconstruction f lives in [0, 1], but the denoisers are calibrated for the classic
[0, 255] intensity convention: NL-Ridge branches on sigma <= 15, TV-Bregman uses
weight = 1/sigma. Feeding them a [0, 1] image with a [0, 255] sigma denoises far too
aggressively; feeding a [0, 1] sigma makes the denoiser a no-op.

So the contract across the whole project is: **sigma is always a [0, 255] noise level**,
and a denoiser is applied as

    D(f * 255, sigma) / 255

as in Zhang et al. (DPIR). For purely spatial denoisers (Gaussian, Bilateral) the scaling
is a harmless no-op. v1 duplicated this bridge in BasePnp and BasePnpAdmm; it lives here
once.

CAUTION — MCMC does not use this bridge. In v1 it called the denoiser on the raw tensor,
with a sigma (default 0.05) that is on f's own scale and doubles as the proposal noise
level. That is inconsistent with PnP, but changing it would silently alter every MCMC
result ever produced, so the v1 behaviour is preserved deliberately. See `Mcmc`.
"""

import torch

from core.features import Feature
from solvers.denoisers import DENOISER_REGISTRY, DENOISER_LIST, ANISOTROPIC_DENOISERS


## denoisers are calibrated for the [0, 255] intensity convention:
DENOISER_SCALE = 255.0

NO_DENOISER = "None"


def resolve(name: str):
    """
    Look a denoiser up by name.

    Returns None for "None" (meaning "no denoising", a legitimate choice), and raises for an
    unknown name — a typo in a config must not silently degrade into running without a prior.
    """
    if name in (NO_DENOISER, None, ""):
        return None
    denoiser = DENOISER_REGISTRY.get(name)
    if denoiser is None:
        known = ", ".join(sorted(DENOISER_REGISTRY))
        raise ValueError(f"Unknown denoiser {name!r}. Available: {known}")
    return denoiser


def denoise(denoiser_fn, f: torch.Tensor, sigma: float, delta: float = 1.0) -> torch.Tensor:
    """Apply a denoiser on its native [0, 255] scale and return the result on f's scale."""
    if denoiser_fn is None:
        return f.clone()
    out = denoiser_fn(f * DENOISER_SCALE, sigma, delta) / DENOISER_SCALE
    return out.reshape(f.shape)


def warn_if_slice_by_slice(report, name: str, data: torch.Tensor) -> None:
    """
    Warn when a 2D-only denoiser is about to be applied to a 3D volume.

    It still works — it is applied plane by plane along z — but it then ignores axial
    correlations entirely, which on an anisotropic problem is a meaningful loss the user
    should know about rather than discover in the results.
    """
    denoiser = DENOISER_REGISTRY.get(name) if name not in (NO_DENOISER, None, "") else None
    if denoiser is None:
        return
    is_3d = data.dim() >= 3 and data.shape[0] > 1
    if is_3d and not getattr(denoiser, "supports_3d", False):
        report(f"[WARNING] Denoiser '{name}' has no native 3D support; "
               f"it will be applied slice by slice along z.")


# ── shared by the denoiser-based solvers ──────────────────────────────────────

## The denoiser, its anisotropy and positivity — the parameters PnP and ADMM-PnP share.
DENOISER_UI_PARAMS = {
    "denoiser": {
        "title": "Denoiser (implicit prior)",
        "type": "option",
        "param_info": {"options_list": DENOISER_LIST},
    },
    "delta": {
        "title": "Anisotropy ratio coefficient",
        "type": "value",
        "requires": {Feature.ANISOTROPIC},
        "depends_on": {"denoiser": ANISOTROPIC_DENOISERS},
        "param_info": {"dtype": float, "unit": "",
                       "latex_name": "\\delta = \\frac{\\Delta z}{\\Delta xy}",
                       "default": 1.0},
    },
    "forced_pos": {
        "title": "Forced positivity",
        "type": "bool",
        "param_info": {"default": True},
    },
}


def relative_noise_level(g: torch.Tensor) -> float:
    """
    The measurement's Gaussian noise std relative to its peak, on the 0-255 scale (0 when g
    is too small to estimate). The scale-free noise level PNPv2 and ADMM-PnPv2 tie their
    denoising to: comparable between inverse problems whatever their intensity scale.
    """
    from core import noise
    try:
        _, b = noise.estimate(g, poisson=False, gaussian=True)
    except ValueError:
        return 0.0
    peak = float(g.max()) or 1.0
    return DENOISER_SCALE * (b ** 0.5) / peak
