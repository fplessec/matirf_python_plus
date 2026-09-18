"""
problems.deconv — the 2D deconvolution inverse problem.

    operator.py   DeconvOperator: convolution by a PSF, applied in Fourier
    problem.py    DECONV, the InverseProblem declaration

Two files, no physics.py: a Gaussian PSF is short enough to live in the operator. This is
what the promise "a new inverse problem is two files" looks like in practice.

    from problems.deconv import DECONV
    prepared = DECONV.prepare(config)
"""

from .operator import DeconvOperator
from .problem import DECONV

__all__ = ["DECONV", "DeconvOperator"]
