"""
core — the problem-agnostic heart of the framework (v2).

Four objects, and the whole framework follows from them:

    Feature          what kind of problem this is (3D? anisotropic? scale-ambiguous?)
    ForwardOperator  the physics: H, H^T, and what can be done with them
    Objective        what is minimized: D(Hf, g) + lambda * R(f)
    InverseProblem   the single declaration tying the three together

The dependency arrows only ever point inward, which is the property v1 lost:

    problems/  ──►  core  ◄──  solvers/
        │                          │
        └──────────►  gui  ◄───────┘

`core` imports nothing from `problems/`, `solvers/` or `gui/`. A solver depends on
`Objective`, never on a problem. A problem depends on `ForwardOperator`, never on a solver.
That is what makes "add an inverse problem without writing an algorithm" possible — and
what makes the reverse, "add an algorithm that works on every problem", possible too.

Read `core/problem.py` first: it states what an inverse problem is in one object, and the
other three files explain the pieces it refers to.
"""

from .features import Feature, NO_FEATURES, features, supports, catalogue
from .operator import ForwardOperator
from .objective import Objective
from .problem import InverseProblem, PreparedProblem, DataMode

__all__ = [
    "Feature", "NO_FEATURES", "features", "supports", "catalogue",
    "ForwardOperator",
    "Objective",
    "InverseProblem", "PreparedProblem", "DataMode",
]
