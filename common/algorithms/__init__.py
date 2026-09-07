"""
ALGORITHMS ARCHITECTURE OVERVIEW (Inverse Problems Framework)

This package implements the full algorithmic stack used by all inverse problem
modules (matirf, deconv, and future extensions).

The system is organized into three layers, from low-level machinery
to full algorithm skeletons.

=====================================================================
1. base/
=====================================================================

Algorithm machinery.

This layer defines the core abstractions used everywhere in the framework:
    > Algorithm (abstract base class for all iterative algorithms)
    > LossComputer (composes a data fidelity + regularization into a loss)
    > DifferentialOperators (spatial derivatives for 2D/3D images)

It is responsible for:
    > defining the algorithmic contract (run, callbacks, threading)
    > composing optimization terms into a loss function
    > providing the mathematical (differential) operators

The optimization terms it composes (DataFidelity, Regularization) are NOT
defined here: they live in reusable/. This layer is intentionally
independent of any specific algorithm or problem. It is the foundation
everything else builds on.

=====================================================================
2. reusable/
=====================================================================

Optimization terms: interfaces + implementations.

This layer owns the mathematical terms the loss operates on. It declares
their interfaces AND ships ready-to-use, domain-independent implementations
that can be plugged into any algorithm without modification:
    > Interfaces (abstract bases): DataFidelity, Regularization
    > Regularizations: L1, L2, Tikhonov, TV, Hessian Frobenius, SHV
    > Data fidelities: Gaussian, Poisson, Poisson-Gaussian

It also exposes the registries (REGULARIZATION_REGISTRY,
DATA_FIDELITY_REGISTRY) linking config.toml <-> UI <-> runtime dispatch.

These components are independent of any specific inverse problem, which
avoids reimplementing the same mathematical objects across modules.

=====================================================================
3. specializable/
=====================================================================

Algorithm skeletons with extension points.

This layer defines full iterative algorithms (Adam, PPXA, ADMM, PnP, MCMC)
that are meant to be subclassed by specific inverse problems.

It handles the heavy lifting:
    > iterative loop and convergence logic
    > loss computation and logging
    > scheduler and stopping criteria
    > UI parameter dictionaries (ui_params)

What it does not define:
    > the forward operator Hf
    > the adjoint operator H^Ttx
    > problem-specific initializations

Instead, it exposes hooks (apply_forward / apply_adjoint, plus
per-algorithm hooks) where subclasses plug in their operators.

Each algorithm also ships a declarative UI-parameter dictionary, accessed
as Base*.ui_params (defined in <algo>/ui_params.py). It describes every
configurable parameter — title, type, dtype, unit, LaTeX name, default —
and drives both the GUI widgets and config.toml. Option lists for the
regularization / data-fidelity fields are pulled from the reusable/
registries, and each parameter carries requires / depends_on flags so it
auto-filters by problem features (e.g. {"3d"} for matirf vs {"2d"} for
deconv).

=====================================================================
=====================================================================

Design summary:
    base          -> algorithm machinery (runner, loss, operators)
    reusable      -> optimization terms (data fidelities, regularizations)
    specializable -> algorithm skeletons with hooks

This structure makes it easy to:
    > add new inverse problems without rewriting the algorithms
    > reuse regularizations and data fidelities across projects
    > keep a consistent architecture across modules
    > avoid duplication of algorithmic logic
"""

from .base import Algorithm, LossComputer
from .specializable import BaseAdam, BaseAdmm, BaseMcmc, BasePnp, BasePnpAdmm, BasePpxa