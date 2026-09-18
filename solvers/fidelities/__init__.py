"""
Data fidelities — the D(Hf, g) term of the objective: how measurement error is penalized.

Choosing one is choosing a NOISE MODEL. It is not a free knob: the right fidelity is the
one matching how the detector actually corrupts the measurement, and using the wrong one
biases the reconstruction no matter how long the solver runs.

Listed by their display names — the strings stored in config.toml and shown in the GUI.

    'L2 (Gaussian noise)'             D = 1/2 ||Hf - g||^2
                                      Additive read noise, independent of intensity. The
                                      usual default, and the one every proximal solver's
                                      hardcoded data step already assumes.

    'KL divergence (Poisson noise)'   D = sum(Hf - g log Hf)
                                      Photon counting: the variance grows with intensity,
                                      so dark pixels are trusted less than bright ones —
                                      the right model for low-light fluorescence.

    'Poisson-Gaussian'                both at once
                                      Photon counting plus read noise, which is what a real
                                      low-light camera actually produces.

Each provides `loss(Hf, g)`; some also provide `prox(...)` for proximal solvers.

ADDING ONE: write a class deriving from `DataFidelity` with `name`, `display_name` and
`loss`, then add it to `_ALL_DATA_FIDELITIES` below. It appears in the GUI and in every
solver that offers a noise model, with no other change anywhere.
"""

from .base import DataFidelity
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
