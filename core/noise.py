"""
Measurement noise — how a simulated measurement is corrupted, one model for every problem.

--------------------------------------------------------------------------------------
The model: Poisson-Gaussian
--------------------------------------------------------------------------------------

A fluorescence camera corrupts its signal in two physically distinct ways, and the standard
model of the field (Foi et al., "Practical Poissonian-Gaussian noise modeling", IEEE TIP
2008) keeps both:

    g_noisy = (scale / N) * Poisson( N * g / scale )  +  sigma * Normal(0, 1)
              \______________ photon noise __________/   \___ read noise ___/

    PHOTON (shot) noise   Light arrives as discrete photons, so a pixel expecting n
                          photons records a Poisson draw of mean n. Its variance EQUALS
                          its mean: noise grows with the signal, and the relative noise
                          1/sqrt(n) is worst in dark regions. N is the number of photons
                          expected at the brightest pixel (scale = max(g)); it is the one
                          parameter that sets how noisy the photon noise is. Few photons,
                          very noisy image; many photons, nearly clean.

    READ noise            The electronics add a signal-independent Gaussian jitter of
                          standard deviation sigma, in the units of g.

Either part can be switched off, which covers the three classical cases:

    photon only    100% Poisson          (sigma off)
    read only      100% Gaussian         (photons off) — what v1 always did
    both           Poisson-Gaussian

--------------------------------------------------------------------------------------
Why the photon count is a parameter at all
--------------------------------------------------------------------------------------

v1 applied `torch.poisson(g)` directly to a measurement normalised to [0, 1]. A Poisson draw
of mean <= 1 is almost always 0 or 1: the "image" became binary speckle. Poisson noise has
no meaning without saying how many photons one unit of signal represents — that is N.
(That branch was also unreachable from the interface: unticking "Gaussian noise" still
produced Gaussian noise.)

--------------------------------------------------------------------------------------
Reproducibility
--------------------------------------------------------------------------------------

Every draw uses its own generator, seeded from the configuration. The same settings always
produce the same noisy measurement, so two algorithms compared on "the same noisy data"
really are — which v1 could not guarantee, since the noise came from the global RNG.
"""

from dataclasses import dataclass
from typing import Optional

import torch

DEFAULT_SEED = 0


@dataclass(frozen=True)
class NoiseModel:
    """
    What the '[add-noise]' section of a config asks for, read once and validated.

    >> photons : float or None   photons at the brightest pixel; None = no photon noise
    >> sigma   : float or None   read-noise standard deviation; None = no read noise
    >> seed    : int             seed of this measurement's noise draw
    """

    photons: Optional[float] = None
    sigma: Optional[float] = None
    seed: int = DEFAULT_SEED

    @property
    def enabled(self) -> bool:
        return self.photons is not None or self.sigma is not None

    def describe(self) -> str:
        parts = []
        if self.photons is not None:
            parts.append(f"Poisson ({self.photons:g} photons at the brightest pixel)")
        if self.sigma is not None:
            parts.append(f"Gaussian (sigma = {self.sigma:g})")
        return " + ".join(parts) + f", seed {self.seed}" if parts else "no noise"


def _number(value):
    """A config value as a float, or None when unset ("None", "null", empty)."""
    if value in (None, "None", "null", ""):
        return None
    return float(value)


def noise_model(params: dict) -> NoiseModel:
    """
    Read the '[add-noise]' section into a NoiseModel.

    Understands the current keys (poisson_noise / photons, gaussian_noise / sigma, seed)
    and v1's (add_noise / is_gaussian / sigma), so an old config.toml keeps working. A v1
    config with noise enabled always produced Gaussian noise — whatever `is_gaussian` said,
    because of the bug described above — so that is what it is mapped to here: the same
    config still gives the same kind of measurement.

    Unset or invalid values do not raise here; `validate` reports them.
    """
    params = params or {}
    seed = params.get("seed", DEFAULT_SEED)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = DEFAULT_SEED

    if "poisson_noise" in params or "gaussian_noise" in params:
        photons = _number(params.get("photons")) if params.get("poisson_noise") else None
        sigma = _number(params.get("sigma")) if params.get("gaussian_noise") else None
        return NoiseModel(photons=photons, sigma=sigma, seed=seed)

    if params.get("add_noise"):                                   # a v1 config
        return NoiseModel(sigma=_number(params.get("sigma")), seed=seed)
    return NoiseModel(seed=seed)


def validate(params: dict) -> list:
    """
    One message per unusable noise setting — the single place this is checked.

    Both problems' `validate` and their preview error messages call this, where v1 repeated
    the same sigma test in four files.
    """
    params = params or {}
    errors = []
    if params.get("poisson_noise"):
        photons = params.get("photons")
        if _number(photons) is None:
            errors.append("Photon noise: the photon count is required when it is enabled")
        elif float(photons) <= 0:
            errors.append(f"Photon noise: the photon count must be positive (got {photons})")
    if params.get("gaussian_noise") or ("gaussian_noise" not in params and params.get("add_noise")):
        sigma = params.get("sigma")
        if _number(sigma) is None:
            errors.append("Read noise: sigma is required when it is enabled")
        elif float(sigma) < 0:
            errors.append(f"Read noise: sigma must be non-negative (got {sigma})")
    return errors


def add_noise_to_measurement(g: torch.Tensor, params: dict) -> torch.Tensor:
    """
    g corrupted as the '[add-noise]' section describes; g itself when no noise is enabled.

    Callers need no `if`: a disabled model returns the input unchanged.
    """
    model = noise_model(params)
    if not model.enabled:
        return g

    generator = torch.Generator(device=g.device).manual_seed(model.seed)
    noisy = g

    if model.photons is not None:
        ## the brightest pixel expects `photons` photons; everything scales from it
        scale = g.max().clamp(min=torch.finfo(g.dtype).tiny)
        expected_counts = g.clamp(min=0) * (model.photons / scale)
        noisy = torch.poisson(expected_counts, generator=generator) * (scale / model.photons)

    if model.sigma is not None and model.sigma > 0:
        noisy = noisy + model.sigma * torch.randn(
            g.shape, generator=generator, dtype=g.dtype, device=g.device)

    return noisy
