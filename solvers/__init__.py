"""
solvers — every reconstruction algorithm, written once for every inverse problem.

A solver receives a `core.Objective` and returns the f that minimizes it. It never sees the
forward operator's internals, never loads data, and never knows which problem it serves —
so there is exactly one implementation of each algorithm, and a new inverse problem
inherits all of them for free.

    SOLVERS["ADAM"]                       -> the class
    SOLVERS["ADAM"].get_ui_params(feats)  -> its parameters, filtered by the problem
    SOLVERS["ADAM"]().solve(objective, f0, params)

`available_for(features)` narrows the registry to the solvers a given problem can actually
run, which is what the GUI lists.

MIGRATION (day 2): Adam is ported. PPXA, ADMM, PnP, ADMM-PnP and MCMC follow on day 3, at
which point `common/algorithms/` and the per-problem `*/algorithms/` packages are deleted.
"""

from .base import Solver
from .adam import Adam

SOLVERS = {cls.name: cls for cls in [
    Adam,
]}


def available_for(problem_features) -> dict:
    """The subset of SOLVERS whose `requires` is satisfied by a problem's features."""
    return {name: cls for name, cls in SOLVERS.items() if cls.supported_by(problem_features)}


__all__ = ["Solver", "Adam", "SOLVERS", "available_for"]
