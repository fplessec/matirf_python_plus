"""
Regularizations — the R(f) term of the objective: what a plausible image looks like.

The measurement alone does not determine f: many images explain g equally well (for
MA-TIRF, H^T H is numerically singular). R(f) breaks that tie by preferring some images
over others, and the choice encodes what you believe about the object.

Listed by their display names — the strings stored in config.toml and shown in the GUI.
Note that "L2 norm" and "L2 norm of the gradient" are NOT the same prior: the first keeps
values small, the second keeps the image smooth.

    'no regularization'              no preference — only for a well-posed problem
    'L1 norm'                        few non-zero pixels: a sparse object
    'L2 norm'                        small values overall; the mildest possible prior
    'L2 norm of the gradient'        smooth everywhere (Tikhonov). Blurs edges along with
                                     the noise, which is its known weakness
    'L1 norm of the gradient'        total variation: piecewise-constant regions with sharp
                                     edges. The classic choice when edges matter more than
                                     texture; tends to flatten gradients into steps
    'frobenius norm of the hessian'  piecewise-LINEAR: smooth ramps without TV's staircasing
    'sparse hessian variation'       hessian regularity plus sparsity, weighted by rho

All of them provide both `loss(f, diff_ops)` and `prox(f, weight, diff_ops)`, so all of
them work with gradient solvers AND proximal ones. A regularization written without a prox
still works with Adam; a proximal solver given one fails at setup with an explicit message
rather than silently producing a wrong answer.

`diff_ops` is a `DifferentialOperators`, which carries the anisotropy ratio delta so an
axial derivative is weighted correctly on a non-cubic grid. Regularizations that use it
declare `uses_diff_ops = True`, which is what puts them in ANISOTROPIC_REGULARIZATIONS and
makes the GUI reveal `delta` only when one of them is selected.

ADDING ONE: write a class deriving from `Regularization` with `name`, `display_name`,
`loss`, and `prox` if you can, then add it to `_ALL_REGULARIZATIONS` below. That single
line makes it available in the GUI and to every solver.
"""

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
