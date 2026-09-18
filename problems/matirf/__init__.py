"""
problems.matirf — the MA-TIRF inverse problem.

    physics.py    the optics: Fresnel transmission, evanescent decay, the H matrix
    operator.py   MatirfOperator, wrapping that physics into the framework's contract
    problem.py    MATIRF, the InverseProblem declaration

Use it as:
    from problems.matirf import MATIRF
    prepared = MATIRF.prepare(config)          # -> operator, g, f_true
"""

from .operator import MatirfOperator
from .problem import MATIRF

__all__ = ["MATIRF", "MatirfOperator"]
