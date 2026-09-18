"""
problems.matirf — the MA-TIRF inverse problem, entirely contained in this folder.

    physics.py    the optics: Fresnel transmission, evanescent decay, the H matrix
    operator.py   MatirfOperator, wrapping that physics into the framework's contract
    problem.py    MATIRF, the InverseProblem declaration
    gui/          this problem's interface
    synthetic/    the reproducible ground-truth generator ('matirf synth')
    data/         measurements and results
    cache/        config.toml

Nothing about MA-TIRF lives anywhere else in the project.

    from problems.matirf import MATIRF
    prepared = MATIRF.prepare(config)
"""

from pathlib import Path

# ── where this problem keeps its files ───────────────────────────────────────
# Defined before the submodule imports below, because they read them.
MATIRF_DIR = Path(__file__).resolve().parent
MATIRF_CACHE_DIR = MATIRF_DIR / 'cache'
MATIRF_CONFIG_PATH = MATIRF_CACHE_DIR / 'config.toml'
MATIRF_DATA_DIR = MATIRF_DIR / 'data'
MATIRF_MEASUREMENTS_DIR = MATIRF_DATA_DIR / 'measurements'
MATIRF_RESULTS_DIR = MATIRF_DATA_DIR / 'results'

## The shape of a fresh config.toml. Sections are left empty on purpose: every reader uses
## .get with a default, so a config emptied by `matirf reset` produces a list of things to
## fill in rather than a KeyError.
DEFAULT_MATIRF_CONFIG = {
    "algorithm": 'None',
    "input-paths": {"mode": "real-data", "tif": "None", "json": "None"},
    "add-noise": {},
    "oper-params": {},
    "algo-params": {},
}

from .operator import MatirfOperator      # noqa: E402
from .problem import MATIRF               # noqa: E402

## the generic name cli.py and the tools look for, whatever the problem is called:
PROBLEM = MATIRF

__all__ = [
    "PROBLEM", "MATIRF", "MatirfOperator",
    "MATIRF_DIR", "MATIRF_CACHE_DIR", "MATIRF_CONFIG_PATH",
    "MATIRF_DATA_DIR", "MATIRF_MEASUREMENTS_DIR", "MATIRF_RESULTS_DIR",
    "DEFAULT_MATIRF_CONFIG",
]
