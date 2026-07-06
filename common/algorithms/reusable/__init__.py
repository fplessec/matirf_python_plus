"""
REUSABLE ALGORITHM LAYER (Prebuilt algorithmic components)

This module provides fully implemented, ready-to-use algorithmic components
for common functionalities across inverse problem applications
(matirf, deconv, and future extensions).

These components are built on top of the BASE layer and implement
concrete mathematical objects that can be used directly without modification.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------

While the BASE layer provides abstract interfaces
(DataFidelity, Regularization),
the REUSABLE layer provides complete, domain-independent implementations
that can be plugged into any algorithm without modification.

---------------------------------------------------------------------
What belongs here
---------------------------------------------------------------------

This layer includes prebuilt algorithmic modules such as:

- Regularizations
    Concrete regularization terms R(f):
        * NoRegularization, L1, L2, Tikhonov
        * TV (Total Variation)
        * Hessian Frobenius, SHV (Sparse Hessian Variation)

- Data Fidelities
    Concrete data fidelity terms D(Hf, g) for different noise models:
        * Gaussian (L2 norm)
        * Poisson (Kullback-Leibler divergence)
        * Poisson-Gaussian (mixed model)

---------------------------------------------------------------------
Design role
---------------------------------------------------------------------

Reusable components:
    - implement the DataFidelity / Regularization interfaces from base/
    - provide loss() and prox() methods for optimization
    - remain independent of specific inverse problem implementations
    - are safe to reuse across matirf, deconv, and future modules

---------------------------------------------------------------------
Key distinction
---------------------------------------------------------------------

- base/:
    abstract algorithmic primitives (Algorithm, DataFidelity, Regularization)

- reusable/:
    concrete implementations (L1, TV, GaussianFidelity, PoissonFidelity)

- specializable/:
    algorithm skeletons requiring problem-specific override (BaseAdam, BasePpxa...)
"""

from .regularizations import (
    Regularization,
    REGULARIZATION_REGISTRY, REGULARIZATION_LIST, ANISOTROPIC_REGULARIZATIONS,
    NoRegularization, L1Regularization, L2Regularization,
    TikhonovRegularization, TVRegularization,
    HessianFrobeniusRegularization, SHVRegularization,
)
from .data_fidelities import (
    DataFidelity,
    DATA_FIDELITY_REGISTRY, DATA_FIDELITY_LIST,
    GaussianFidelity, PoissonFidelity, PoissonGaussianFidelity,
)
