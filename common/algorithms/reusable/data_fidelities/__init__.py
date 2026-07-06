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
