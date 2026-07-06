from .base import Regularization
from .no_reg import NoRegularization
from .l1 import L1Regularization
from .l2 import L2Regularization
from .tikhonov import TikhonovRegularization
from .tv import TVRegularization
from .hessian_frobenius import HessianFrobeniusRegularization
from .shv import SHVRegularization


_ALL_REGULARIZATIONS = [
    NoRegularization,
    L1Regularization,
    L2Regularization,
    TikhonovRegularization,
    TVRegularization,
    HessianFrobeniusRegularization,
    SHVRegularization,
]

# Registry maps display_name -> class (primary key, used by UI and config.toml).
# Also maps name -> class (fallback, for backward compatibility).
REGULARIZATION_REGISTRY = {}
for cls in _ALL_REGULARIZATIONS:
    REGULARIZATION_REGISTRY[cls.display_name] = cls
    REGULARIZATION_REGISTRY[cls.name] = cls

# List for UI options (display_name only).
REGULARIZATION_LIST = [cls.display_name for cls in _ALL_REGULARIZATIONS]

# Regularizations that use differential operators (benefit from delta in 3D anisotropic).
ANISOTROPIC_REGULARIZATIONS = [cls.display_name for cls in _ALL_REGULARIZATIONS if cls.uses_diff_ops]
