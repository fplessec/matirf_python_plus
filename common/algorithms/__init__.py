"""
ALGORITHMS ARCHITECTURE OVERVIEW (Inverse Problems Framework)

This package implements the full algorithmic stack used by all inverse problem
modules (matirf, deconv, and future extensions such as super-resolution).

The system is organized into three layers, from low-level primitives
to full algorithm skeletons.

=====================================================================
1. base/
=====================================================================

Low-level building blocks for iterative algorithms.

This layer defines the core abstractions used everywhere in the framework:
    - Algorithm (abstract base class for all iterative algorithms)
    - LossComputer (assembles data fidelity + regularization into a loss)
    - DataFidelity (abstract base for noise models)
    - Regularization (abstract base for regularization terms)
    - DifferentialOperators (spatial derivatives for 2D/3D images)

It is responsible for:
    - defining the algorithmic contract (run, callbacks, threading)
    - composing optimization terms into a loss function
    - providing mathematical operators

This layer is intentionally independent of any specific algorithm or problem.
It is the foundation everything else builds on.

=====================================================================
2. specializable/
=====================================================================

Algorithm skeletons with extension points.

This layer defines full iterative algorithms (Adam, PPXA, ADMM, PnP, MCMC)
that are meant to be subclassed by specific inverse problems.

It handles the heavy lifting:
    - iterative loop and convergence logic
    - loss computation and logging
    - scheduler and stopping criteria
    - UI parameter dictionaries (ui_params)

What it does not define:
    - the forward operator Hf
    - the adjoint operator H^T x
    - problem-specific initializations

Instead, it exposes hooks where subclasses plug in their operators.

Think of it as a working algorithm shell that becomes concrete
only when specialized with a forward model.

=====================================================================
3. reusable/
=====================================================================

Prebuilt algorithmic components at the mathematical level.

This layer contains ready-to-use implementations that can be plugged
into any algorithm without modification.

Typical examples include:
    - regularizations (L1, L2, TV, Tikhonov, Hessian Frobenius, SHV)
    - data fidelities (Gaussian, Poisson, Poisson-Gaussian)

These components implement the interfaces from base/ and are designed
to be independent of any specific inverse problem.

They help avoid reimplementing the same mathematical objects across modules.

=====================================================================

Design summary:

    base          -> algorithmic primitives
    specializable -> algorithm skeletons with hooks
    reusable      -> concrete mathematical components

This structure makes it easy to:
    - add new inverse problems without rewriting the algorithms
    - reuse regularizations and data fidelities across projects
    - keep a consistent architecture across modules
    - avoid duplication of algorithmic logic
"""

from .base import Algorithm, LossComputer, DifferentialOperators
from .specializable import (
    BaseAdam, ADAM_UI_PARAMETERS,
    BaseAdmm, ADMM_UI_PARAMETERS,
    BaseMcmc, MCMC_UI_PARAMETERS,
    BasePnp, PNP_UI_PARAMETERS,
    BasePpxa, PPXA_UI_PARAMETERS,
)


ALGORITHM_REGISTRY = {cls.name: cls for cls in [
    BaseMcmc,
    BaseAdam,
    BasePnp,
    BasePpxa,
    BaseAdmm,
]}
