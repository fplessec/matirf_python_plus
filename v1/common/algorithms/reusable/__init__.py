"""
REUSABLE ALGORITHM LAYER (Optimization terms: interfaces + implementations)

This module defines the optimization terms plugged into any algorithm's
loss (matirf, deconv, and future extensions). It provides BOTH the
abstract interfaces (DataFidelity, Regularization) AND their ready-to-use
concrete implementations.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------
The BASE layer defines the algorithm machinery (Algorithm, LossComputer,
DifferentialOperators). This REUSABLE layer owns the mathematical terms
that machinery operates on: it declares their interfaces and ships
complete, domain-independent implementations that can be plugged in
without modification.

---------------------------------------------------------------------
What belongs here
---------------------------------------------------------------------
> Interfaces (abstract bases, defined in this layer):
    > DataFidelity   -> contract for data terms   D(Hf, g)
    > Regularization -> contract for prior terms  R(f)

> Regularizations (concrete R(f)):
    > NoRegularization, L1, L2
    > Tikhonov, TV (Total Variation)
    > Hessian Frobenius, SHV (Sparse Hessian Variation)

> Data Fidelities (concrete D(Hf, g), one per noise model):
    > Gaussian (L2 norm)
    > Poisson (Kullback-Leibler divergence)
    > Poisson-Gaussian (mixed model)

---------------------------------------------------------------------
Registry
---------------------------------------------------------------------
Single source of truth linking config.toml <-> UI <-> runtime dispatch:
    > REGULARIZATION_REGISTRY / DATA_FIDELITY_REGISTRY
        > map display_name -> class (primary key)
        > also map name -> class (backward-compatible fallback)
    > REGULARIZATION_LIST / DATA_FIDELITY_LIST
        > display names for UI option widgets
    > ANISOTROPIC_REGULARIZATIONS
        > regularizations that use differential operators (delta in 3D)

---------------------------------------------------------------------
---------------------------------------------------------------------
Overall design philosophy of Reusable terms:
    > define AND implement the DataFidelity / Regularization interfaces
    > provide loss() and prox() methods for optimization
    > remain independent of specific inverse problem implementations
    > are safe to reuse across matirf, deconv, and future modules
"""

from .regularizations import (
    Regularization,
    REGULARIZATION_REGISTRY, REGULARIZATION_LIST, ANISOTROPIC_REGULARIZATIONS,
)
from .data_fidelities import (
    DataFidelity,
    DATA_FIDELITY_REGISTRY, DATA_FIDELITY_LIST,
)
