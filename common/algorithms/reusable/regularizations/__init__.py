"""
MIGRATION SHIM (v1 -> v2). The regularizations now live in `solvers/regularizers/`.

This module only re-exports them so the v1 algorithm stack keeps running while both
versions coexist. It is deleted together with `common/algorithms/`. New code imports from
`solvers.regularizers`.
"""

from solvers.regularizers import *          # noqa: F401,F403
from solvers.regularizers import (          # noqa: F401
    Regularization, NoRegularization, REGULARIZATION_REGISTRY, REGULARIZATION_LIST,
    ANISOTROPIC_REGULARIZATIONS,
)
