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

All six algorithms are ported (day 3). Together they replace `common/algorithms/` and both
per-problem `*/algorithms/` packages — about 470 lines of pure plumbing — which are deleted
once matirf and deconv are ported onto `core` (day 5).

The unlock was `ForwardOperator.solve_normal`: PPXA, ADMM, PnP and PnP-ADMM were
problem-specific in v1 only because each needed to invert (H^T H + lam I) its own way, and
MCMC only because its data-consistency step needed a ridge inverse. One primitive on the
operator removed every one of those subclasses.
"""

from .base import Solver
from .adam import Adam
from .ppxa import Ppxa
from .admm import Admm
from .pnp import Pnp
from .pnp_admm import PnpAdmm
from .mcmc import Mcmc

## One solver per algorithm — after the benchmark, the weaker of each v1/v2 pair was retired
## to solvers/old/ (see docs/algorithms/): ADMM and ADMM-PnP kept their v1, PnP and MCMC their
## calibrated (former "v2") version. Registration order is the order the GUI lists them in.
SOLVERS = {cls.name: cls for cls in [
    Adam,
    Ppxa,
    Admm,
    Pnp,
    PnpAdmm,
    Mcmc,
]}


def available_for(problem_features) -> dict:
    """The subset of SOLVERS whose `requires` is satisfied by a problem's features."""
    return {name: cls for name, cls in SOLVERS.items() if cls.supported_by(problem_features)}


__all__ = ["Solver", "Adam", "Ppxa", "Admm", "Pnp", "PnpAdmm", "Mcmc", "SOLVERS", "available_for"]
