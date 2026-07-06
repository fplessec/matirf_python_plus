"""
BASE ALGORITHM LAYER (Algorithmic primitives)

This module defines the lowest-level building blocks used to construct
iterative algorithms for inverse problems.

It implements the core abstractions that all algorithms rely on:

---------------------------------------------------------------------
1. Algorithm
---------------------------------------------------------------------
Abstract base class for all iterative algorithms.

It is responsible for:
    - running the algorithm in a background thread
    - communicating with the pipeline via callbacks
    - providing interruption and reproducibility mechanisms
    - factory helpers for creating data fidelity, regularization, loss

This is the atomic algorithmic unit of the system.

---------------------------------------------------------------------
2. LossComputer
---------------------------------------------------------------------
Assembles a data fidelity term and a regularization term into
a single callable loss function:

    L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f)

This is the core composition utility for gradient-based algorithms.

---------------------------------------------------------------------
3. DataFidelity
---------------------------------------------------------------------
Abstract base class for data fidelity terms D(Hf, g).

Each noise model (Gaussian, Poisson, ...) provides:
    - loss(Hf, g)   : value of the data fidelity
    - prox(...)      : proximal operator

---------------------------------------------------------------------
4. Regularization
---------------------------------------------------------------------
Abstract base class for regularization terms R(f).

Each regularization (L1, TV, ...) provides:
    - loss(f, diff_ops)          : value of the regularization
    - prox(f, lambda, diff_ops)  : proximal operator

---------------------------------------------------------------------
5. DifferentialOperators
---------------------------------------------------------------------
Spatial differential operators (gradient, divergence, laplacian, hessian)
for 2D and 3D images, with anisotropy support.

Used by regularizations that involve spatial derivatives.

---------------------------------------------------------------------

Overall design philosophy:
    - each class is independent of any specific algorithm or problem
    - Algorithm is the abstract runner, LossComputer composes the pieces
    - DataFidelity and Regularization define the optimization terms
    - DifferentialOperators provides the mathematical tools

This module is framework-agnostic with respect to inverse problems:
it only defines generic algorithmic primitives.
"""

from .algorithm import Algorithm
from .loss_computer import LossComputer
from .differential_operators import DifferentialOperators
