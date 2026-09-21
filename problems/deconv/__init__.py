"""
problems.deconv — the 2D deconvolution inverse problem, entirely contained in this folder.

    operator.py   DeconvOperator: convolution by a PSF, applied in Fourier
    problem.py    DECONV, the InverseProblem declaration
    gui/          this problem's interface
    data/         measurements and results
    cache/        config.toml

No physics.py: a Gaussian PSF is short enough to live in the operator. This is what the
promise "a new inverse problem is a handful of files" looks like in practice.

    from problems.deconv import DECONV
    prepared = DECONV.prepare(config)
"""

from pathlib import Path

DECONV_DIR = Path(__file__).resolve().parent
DECONV_CACHE_DIR = DECONV_DIR / 'cache'
DECONV_CONFIG_PATH = DECONV_CACHE_DIR / 'config.toml'
DECONV_DATA_DIR = DECONV_DIR / 'data'
DECONV_MEASUREMENTS_DIR = DECONV_DATA_DIR / 'measurements'
DECONV_RESULTS_DIR = DECONV_DATA_DIR / 'results'

DEFAULT_DECONV_CONFIG = {
    "algorithm": 'None',
    "input-paths": {"mode": "synthetic-data", "png": "None", "json": "None",
                    "normalization": "peak"},
    "add-noise": {},
    "noise-model": {},
    "algo-params": {},
}

from .operator import DeconvOperator      # noqa: E402
from .problem import DECONV               # noqa: E402

## the generic name cli.py and the tools look for, whatever the problem is called:
PROBLEM = DECONV

__all__ = [
    "PROBLEM", "DECONV", "DeconvOperator",
    "DECONV_DIR", "DECONV_CACHE_DIR", "DECONV_CONFIG_PATH",
    "DECONV_DATA_DIR", "DECONV_MEASUREMENTS_DIR", "DECONV_RESULTS_DIR",
    "DEFAULT_DECONV_CONFIG",
]
