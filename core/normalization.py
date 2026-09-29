"""
Normalization — putting a measurement on the common scale every setting is expressed in.

--------------------------------------------------------------------------------------
Why it matters for the whole formulation
--------------------------------------------------------------------------------------

Every quantity that describes the data is expressed in NORMALIZED units: the noise level
(a, b in core/noise.py), the data fidelity D, and therefore lambda_reg. Choosing the
normalization is choosing that unit — which is why it is a setting of the input files, next
to the measurement it applies to, and why "See preprocessed file" shows its effect.

Two properties are kept by every method:

    NON-NEGATIVE   the output is >= 0. The framework solves g = H f with f >= 0 and g >= 0,
                   so a negative value left by background subtraction is set to 0 first.

    A PURE SCALE   except min-max, every method DIVIDES by a constant. g = H f is linear:
                   g / s = H (f / s), so dividing changes the unit and nothing else. Min-max
                   also subtracts the minimum, which turns g = H f into g = (H f - c) / s —
                   an offset no non-negative f can explain. It is offered because v1 used it,
                   but the default is "peak".

--------------------------------------------------------------------------------------
The methods
--------------------------------------------------------------------------------------

    peak             g / max(g)                    in [0, 1]; the default
    percentile 99.9  min(g / q99.9(g), 1)          in [0, 1]; ignores a few hot pixels,
                                                   which saturate at 1
    min-max          (g - min) / (max - min)       in [0, 1]; adds an offset (see above)
    L2 energy        g / ||g||_2                   unit energy (v1 mode 2)
    RMS              g / sqrt(mean(g^2))           unit root-mean-square (v1 mode 3)
    mean             g / mean(g)                   unit mean (v1 mode 4)
    none             g                             the file as it is (clipped at 0)

The first three give intensities in [0, 1], which is what the noise parameters are designed
for: at "peak", a read-noise sigma of 0.01 is 1 % of the brightest pixel.
"""

import torch

from core import noise

DEFAULT_NORMALIZATION = "peak"

## name -> the latex shown under the choice in the interface
FORMULAS = {
    "peak": r"g \leftarrow g\ /\ \max(g)",
    "percentile 99.9": r"g \leftarrow \min\left(g\ /\ q_{99.9}(g),\ 1\right)",
    "min-max": r"g \leftarrow (g - \min g)\ /\ (\max g - \min g)",
    "L2 energy": r"g \leftarrow g\ /\ \|g\|_2",
    "RMS": r"g \leftarrow g\ /\ \sqrt{\overline{g^2}}",
    "mean": r"g \leftarrow g\ /\ \overline{g}",
    "none": r"g \leftarrow g",
}
NORMALIZATIONS = list(FORMULAS)

## v1 stored an integer (matirf/settings.py `normalization`); an old config keeps working
_V1_CODES = {0: "none", 1: "min-max", 2: "L2 energy", 3: "RMS", 4: "mean"}

## torch.quantile refuses inputs above 2^24 elements; a regular subsample is plenty
_QUANTILE_MAX_SIZE = 2 ** 24


def resolve(name) -> str:
    """The method a config value designates (a name, a v1 integer, or unset -> default)."""
    if name in (None, "", "None", "null"):
        return DEFAULT_NORMALIZATION
    if isinstance(name, int) and not isinstance(name, bool):
        return _V1_CODES.get(name, DEFAULT_NORMALIZATION)
    return name


def validate(name) -> list:
    method = resolve(name)
    if method not in FORMULAS:
        return [f"Normalization: unknown method {name!r} "
                f"(choose one of: {', '.join(NORMALIZATIONS)})"]
    return []


def formula(name) -> str:
    return FORMULAS.get(resolve(name), r"\text{unknown normalization}")


def normalize(g: torch.Tensor, name=DEFAULT_NORMALIZATION) -> torch.Tensor:
    """g >= 0, on the scale `name` describes. An all-zero input stays all zero."""
    method = resolve(name)
    if method not in FORMULAS:
        raise ValueError(validate(name)[0])

    g = g.clamp(min=0)
    if method == "none":
        return g
    if method == "min-max":
        low, high = g.min(), g.max()
        return (g - low) / (high - low) if high > low else torch.zeros_like(g)

    if method == "peak":
        scale = g.max()
    elif method == "percentile 99.9":
        scale = _quantile(g, 0.999)
    elif method == "L2 energy":
        scale = g.square().sum().sqrt()
    elif method == "RMS":
        scale = g.square().mean().sqrt()
    else:                                            # "mean"
        scale = g.mean()

    if scale <= 0:
        return torch.zeros_like(g)
    g = g / scale
    return g.clamp(max=1.0) if method == "percentile 99.9" else g


def _quantile(g: torch.Tensor, q: float) -> torch.Tensor:
    flat = g.flatten()
    if flat.numel() > _QUANTILE_MAX_SIZE:
        flat = flat[:: flat.numel() // _QUANTILE_MAX_SIZE + 1]
    return torch.quantile(flat, q)


# ── the last two steps of every preprocessing ─────────────────────────────────

def normalize_and_add_noise(g: torch.Tensor, config: dict) -> torch.Tensor:
    """
    What every measurement goes through last, real or simulated, in every problem:

        1. normalization, chosen in '[input-paths] normalization'
        2. the simulated noise of '[add-noise]' (nothing when it is disabled), clipped at 0

    In that order, and it matters: the noise parameters are expressed in normalized units
    (a sigma of 0.01 is 1 % of an intensity of 1), so the unit must be fixed before the
    noise is drawn. Written once here so that no problem can do it in another order.
    """
    g = normalize(g, config.get("input-paths", {}).get("normalization"))
    return noise.add_noise_to_measurement(g, config.get("add-noise", {}))
