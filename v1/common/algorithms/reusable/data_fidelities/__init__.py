"""
MIGRATION SHIM (v1 -> v2). The data fidelities now live in `solvers/fidelities/`.

This module only re-exports them so the v1 algorithm stack keeps running while both
versions coexist. It is deleted together with `common/algorithms/`. New code imports from
`solvers.fidelities`.
"""

from solvers.fidelities import *            # noqa: F401,F403
from solvers.fidelities import (            # noqa: F401
    DataFidelity, GaussianFidelity, PoissonFidelity, PoissonGaussianFidelity,
    DATA_FIDELITY_REGISTRY, DATA_FIDELITY_LIST,
)
