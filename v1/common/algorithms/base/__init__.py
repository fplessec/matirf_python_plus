"""
BASE ALGORITHM LAYER (Algorithmic primitives)

This module defines the lowest-level building blocks used to construct
iterative algorithms for inverse problems (matirf, deconv, and future
extensions).

It defines exactly three components: Algorithm, LossComputer, and
DifferentialOperators.

---------------------------------------------------------------------
1. Algorithm
---------------------------------------------------------------------
Abstract base class for all iterative algorithms.

It is responsible for:
    > running the algorithm in a background thread
    > communicating with the pipeline via callbacks
    > providing interruption and reproducibility mechanisms
    > factory helpers that assemble a loss from:
        > a data fidelity (pulled from reusable/)
        > a regularization (pulled from reusable/)

This is the atomic algorithmic unit of the system.

---------------------------------------------------------------------
2. LossComputer
---------------------------------------------------------------------
Assembles a data fidelity term and a regularization term into a single
callable loss function:

    L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

It only composes the terms: D and R are passed in as objects (defined
in reusable/). This is the core composition utility for gradient-based
algorithms.

---------------------------------------------------------------------
3. DifferentialOperators
---------------------------------------------------------------------
Spatial differential operators (gradient, divergence, laplacian,
hessian) for 2D and 3D images, with anisotropy support.

Used by regularizations that involve spatial derivatives.

---------------------------------------------------------------------
---------------------------------------------------------------------
Overall design philosophy:
    > Algorithm is the abstract runner
    > LossComputer composes the optimization terms
    > DifferentialOperators provides the mathematical tools

This module is framework-agnostic with respect to inverse problems:
it only defines generic algorithmic primitives.
"""

from .algorithm import Algorithm
from .loss_computer import LossComputer
from .differential_operators import DifferentialOperators
