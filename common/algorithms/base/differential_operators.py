"""
MIGRATION SHIM (v1 -> v2). DifferentialOperators now lives in
`solvers/differential_operators.py`.

Re-exported here so the v1 algorithm stack keeps running while both versions coexist.
Deleted together with `common/algorithms/`. New code imports from
`solvers.differential_operators`.
"""

from solvers.differential_operators import DifferentialOperators   # noqa: F401
