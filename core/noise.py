"""
Measurement noise — one model for every problem: how a measurement is corrupted, how the
corruption is described, and how its level is recovered from the data.

--------------------------------------------------------------------------------------
The model: Poisson-Gaussian, in normalized units
--------------------------------------------------------------------------------------

A fluorescence camera corrupts its signal in two physically distinct ways, and the standard
model of the field (Foi et al., "Practical Poissonian-Gaussian noise modeling", IEEE TIP
2008) keeps both. Written in the NORMALIZED units of the measurement (core/normalization.py),
it has exactly two parameters:

    g_noisy = max( a * Poisson(g / a)  +  sqrt(b) * Normal(0, 1) ,  0 )
                   \_ photon noise _/     \____ read noise ____/

    Var(g_noisy) = a * g + b

    a = 1 / N    PHOTON (shot) noise. Light arrives as discrete photons: a pixel of
                 normalized intensity g collects g * N photons on average, and a Poisson
                 draw has variance equal to its mean. N is the number of photons for an
                 intensity of 1 — the brightest pixel under "peak" normalization. Few
                 photons, very noisy image; the relative noise 1/sqrt(g N) is worst in the
                 dark regions.

    b = sigma^2  READ noise. The electronics add a signal-independent Gaussian jitter of
                 standard deviation sigma, in normalized units: sigma = 0.01 is 1 % of an
                 intensity of 1.

Either part can be switched off, which covers the three classical cases:

    photon only    100% Poisson          b = 0
    read only      100% Gaussian         a = 0   — what v1 always did
    both           Poisson-Gaussian

The SAME two numbers parameterize the data fidelity D (solvers/fidelities): the noise that
corrupts a simulated measurement and the noise a reconstruction assumes are one object,
written once. That is what lets lambda_reg mean "how much prior" and nothing else.

--------------------------------------------------------------------------------------
Non-negativity
--------------------------------------------------------------------------------------

The framework solves g = H f with f >= 0 and g >= 0, so the noisy measurement is clipped
at 0. A Poisson draw is never negative; only read noise on a dark pixel can be. The price is
a small, documented bias on dark pixels (a clipped Normal(0, sigma) has mean ~0.4 sigma),
and `estimate` accordingly ignores clipped pixels.

--------------------------------------------------------------------------------------
Estimation
--------------------------------------------------------------------------------------

`estimate(g)` recovers (a, b) from a measurement, real or simulated. It uses the local
noise of 2x2 blocks: their mixed second difference cancels a smooth signal and keeps the
noise, whose variance is then regressed against the block mean — Var = a * mean + b.

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
import torch.nn.functional as F

DEFAULT_SEED = 0


@dataclass(frozen=True)
class NoiseModel:
    """
    What the '[add-noise]' section of a config asks for, read once and validated.

    >> photons : float or None   photons for an intensity of 1; None = no photon noise
    >> sigma   : float or None   read-noise standard deviation; None = no read noise
    >> seed    : int             seed of this measurement's noise draw
    """

    photons: Optional[float] = None
    sigma: Optional[float] = None
    seed: int = DEFAULT_SEED

    @property
    def enabled(self) -> bool:
        return self.photons is not None or self.sigma is not None

    @property
    def a(self) -> float:
        """The photon-noise parameter, 1 / N (0 without photon noise)."""
        return 1.0 / self.photons if self.photons else 0.0

    @property
    def b(self) -> float:
        """The read-noise parameter, sigma^2 (0 without read noise)."""
        return self.sigma ** 2 if self.sigma else 0.0

    def describe(self) -> str:
        parts = []
        if self.photons is not None:
            parts.append(f"Poisson ({self.photons:g} photons for an intensity of 1)")
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
    g corrupted as the '[add-noise]' section describes, clipped at 0; g itself when no noise
    is enabled.

    Callers need no `if`: a disabled model returns the input unchanged.
    """
    model = noise_model(params)
    if not model.enabled:
        return g

    generator = torch.Generator(device=g.device).manual_seed(model.seed)
    noisy = g.clamp(min=0)

    if model.photons is not None:
        ## a pixel of intensity g collects g * N photons on average
        noisy = torch.poisson(noisy * model.photons, generator=generator) / model.photons

    if model.sigma is not None and model.sigma > 0:
        noisy = noisy + model.sigma * torch.randn(
            g.shape, generator=generator, dtype=g.dtype, device=g.device)

    return noisy.clamp(min=0)


# ── the model, written out ────────────────────────────────────────────────────

def formula(params: dict) -> str:
    """
    The noise the '[add-noise]' section adds, as a latex line with the values of a and b.

    Shown at the bottom of that section: it changes with the ticked boxes, so the user sees
    the exact model — purely Gaussian, purely Poisson, or both — and not just its settings.
    """
    if validate(params):
        return r"\text{incomplete noise settings}"
    model = noise_model(params)
    if model.photons is not None and model.sigma:
        return (r"g_{noisy} = \max\left(a\,\mathcal{P}(g/a) + \sqrt{b}\,\varepsilon,\ 0\right)"
                rf",\quad a = 1/N = {_latex_number(model.a)},\ b = \sigma^2 = {_latex_number(model.b)}")
    if model.photons is not None:
        return (r"g_{noisy} = a\,\mathcal{P}(g/a)"
                rf",\quad a = 1/N = {_latex_number(model.a)},\ b = 0")
    if model.sigma:
        return (r"g_{noisy} = \max\left(g + \sqrt{b}\,\varepsilon,\ 0\right)"
                rf",\quad a = 0,\ b = \sigma^2 = {_latex_number(model.b)}")
    return r"g_{noisy} = g\quad(\text{no noise added})"


def _latex_number(x: float) -> str:
    return f"{x:.3g}"


# ── estimation ────────────────────────────────────────────────────────────────

## Immerkaer's mask: the difference of two Laplacians, normalized so that white noise keeps
## its variance (the squared coefficients sum to 36, hence the / 6)
_IMMERKAER = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]],
                          dtype=torch.float64).view(1, 1, 3, 3) / 6
## neighbourhoods per variance bin, and bins: a robust variance in each, enough bins to fit
_BINS = 24
_MIN_BLOCKS_PER_BIN = 40
## the MAD of a Normal variable is 0.6745 sigma
_MAD_TO_STD = 1.4826


def estimate(g: torch.Tensor, poisson: bool = True, gaussian: bool = True) -> tuple:
    """
    (a, b) such that Var(g) ~ a * g + b, estimated from g itself.

    `poisson` / `gaussian` say which parts the model has: the missing one is returned as 0
    and not fitted. Works on any image or stack of images: the noise is measured within the
    last two dimensions, where neighbouring pixels see nearly the same signal.

    The method, in three steps:
        1. every 3x3 neighbourhood is filtered by Immerkaer's mask (Immerkaer, "Fast noise
           variance estimation", CVIU 1996) — [1 -2 1; -2 4 -2; 1 -2 1] / 6 — which cancels
           any locally quadratic signal and keeps the noise with its variance unchanged;
           m = the neighbourhood mean
        2. neighbourhoods sorted by m into equal-count bins; in each, a robust (MAD)
           variance of the filtered values
        3. a weighted least-squares fit of that variance against m
    Neighbourhoods touching a clipped (zero) pixel are ignored, and so are the dark bins
    where the clipping at 0 would still bias the variance (mean below two standard
    deviations).

    Accuracy: within ~10-30 % on blurred measurements (g = H f is smooth, which is the case
    this framework meets). On a sharply textured image the texture reads as noise and the
    estimate is biased upward — set the parameters by hand there.
    """
    if not (poisson or gaussian):
        return 0.0, 0.0
    x = g.detach().to(torch.float64)
    if x.dim() < 2 or min(x.shape[-2:]) < 3:
        raise ValueError("noise estimation needs images of at least 3x3 pixels")
    x = x.reshape(-1, 1, *x.shape[-2:])
    d = F.conv2d(x, _IMMERKAER.to(x.device)).flatten()
    m = F.avg_pool2d(x, 3, stride=1).flatten()
    unclipped = (-F.max_pool2d(-x, 3, stride=1)).flatten() > 0
    d, m = d[unclipped], m[unclipped]
    if d.numel() < 2 * _MIN_BLOCKS_PER_BIN:
        raise ValueError(f"too few unclipped pixels to estimate the noise ({d.numel()})")

    n_bins = max(1, min(_BINS, d.numel() // _MIN_BLOCKS_PER_BIN))
    order = torch.argsort(m)
    means, variances = [], []
    for chunk in torch.chunk(order, n_bins):
        dc = d[chunk]
        means.append(float(m[chunk].mean()))
        variances.append(float((_MAD_TO_STD * (dc - dc.median()).abs().median()) ** 2))
    means = torch.tensor(means, dtype=torch.float64)
    variances = torch.tensor(variances, dtype=torch.float64)
    if float(variances.max()) <= 0:                  # a noiseless (or quantized) image
        return 0.0, 0.0

    reliable = means > 2 * variances.sqrt()
    if int(reliable.sum()) >= 3:
        means, variances = means[reliable], variances[reliable]
    ## a variance estimate is itself noisy in proportion to its size: weight by 1 / v^2
    weights = 1.0 / (variances + 1e-6 * float(variances.max())) ** 2
    weights = weights / weights.max()           # only their ratios matter; keeps lstsq finite

    if not poisson:
        return 0.0, float(variances.median())
    if not gaussian:
        return _fit_through_origin(means, variances, weights), 0.0
    a, b = _fit_line(means, variances, weights)
    if a < 0:
        return 0.0, float((weights * variances).sum() / weights.sum())
    if b < 0:
        return _fit_through_origin(means, variances, weights), 0.0
    return a, b


def _fit_through_origin(m, v, w) -> float:
    return max(0.0, float((w * m * v).sum() / (w * m * m).sum()))


def _fit_line(m, v, w) -> tuple:
    """Weighted least squares v = a m + b."""
    design = torch.stack([m, torch.ones_like(m)], dim=1) * w.sqrt().unsqueeze(1)
    target = (v * w.sqrt()).unsqueeze(1)
    solution = torch.linalg.lstsq(design, target).solution.flatten()
    return float(solution[0]), float(solution[1])
