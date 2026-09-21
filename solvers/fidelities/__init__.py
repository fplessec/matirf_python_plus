"""
Data fidelities — the D(Hf, g) term of the objective: the noise model, as a likelihood.

Choosing one is choosing a NOISE MODEL, and that choice belongs to the measurement, not to
any algorithm: it is made in the '[noise-model]' section of the config, and each solver
declares which models it can handle (`Solver.supported_noise_models`).

Every fidelity is the per-pixel negative log-likelihood of its noise, parameterized by the
(a, b) of core/noise.py — Var(g) = a * Hf + b — see base.py for why that makes lambda_reg a
pure prior weight.

    'L2 (Gaussian noise)'             D = mean((Hf - g)^2) / (2 b)             b = sigma^2
                                      Additive read noise, independent of intensity. The
                                      one every splitting solver's data step assumes.

    'KL divergence (Poisson noise)'   D = mean(Hf - g + g log(g / Hf)) / a     a = 1 / N
                                      Photon counting: the variance grows with intensity,
                                      so bright pixels are trusted less in absolute terms.

    'Poisson-Gaussian'                both at once, Var = a Hf + b
                                      Photon counting plus read noise, which is what a real
                                      low-light camera actually produces.

ADDING ONE: write a class deriving from `DataFidelity` with `name`, `display_name`,
`noise_parameters`, `formula` and `loss`, then add it to `_ALL_DATA_FIDELITIES` below. It
appears in the interface's noise-model section with no other change anywhere.
"""

from .base import DataFidelity, NOISE_FLOOR
from .gaussian import GaussianFidelity
from .poisson import PoissonFidelity
from .poisson_gaussian import PoissonGaussianFidelity


_ALL_DATA_FIDELITIES = [
    GaussianFidelity,
    PoissonFidelity,
    PoissonGaussianFidelity,
]

# Registry maps display_name -> class (primary key, used by UI and config.toml).
# Also maps name -> class (fallback, for backward compatibility).
DATA_FIDELITY_REGISTRY = {}
for cls in _ALL_DATA_FIDELITIES:
    DATA_FIDELITY_REGISTRY[cls.display_name] = cls
    DATA_FIDELITY_REGISTRY[cls.name] = cls

# List for UI options (display_name only).
DATA_FIDELITY_LIST = [cls.display_name for cls in _ALL_DATA_FIDELITIES]

# The short names solvers use in `supported_noise_models`.
NOISE_MODEL_NAMES = frozenset(cls.name for cls in _ALL_DATA_FIDELITIES)
