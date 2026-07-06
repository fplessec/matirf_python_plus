"""
SPECIALIZABLE ALGORITHM LAYER (Extensible algorithm framework)

This module defines partially implemented iterative algorithms
that form the computational backbone of inverse problem solvers
(matirf, deconv, and future extensions).

Unlike the BASE layer (algorithmic primitives) and the REUSABLE layer
(concrete regularizations and data fidelities), this layer defines
full algorithm skeletons with explicit extension points ("hooks")
that must be implemented by problem-specific modules.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------

The SPECIALIZABLE layer provides complete algorithm implementations
that are functional but not fully defined.

Each algorithm:
    - implements the full iterative loop and convergence logic
    - defines a set of required hooks for customization
    - enforces architectural consistency across problems
    - carries its own ui_params dictionary for GUI integration

---------------------------------------------------------------------
Main components
---------------------------------------------------------------------

1. BaseAdam
------------------------------------------------
Gradient-based optimizer using torch.optim.Adam.

Extension points:
    - apply_forward(H, f)   : computes Hf
    - apply_adjoint(H, x)   : computes H^T x
    - init_f(g, H, params)  : initialization of f (default: g.clone())

2. BasePpxa
------------------------------------------------
Parallel Proximal Algorithm (PPXA).

Extension points:
    - apply_forward / apply_adjoint
    - init_f / precompute
    - compute_data_prox(u, params) : proximal operator for data fidelity

3. BaseAdmm
------------------------------------------------
Alternating Direction Method of Multipliers.

Extension points:
    - apply_forward / apply_adjoint
    - init_f / precompute
    - solve_linear_system(Htg, v, mu, params)
    - compute_threshold(g, H, params)

4. BasePnp
------------------------------------------------
Plug-and-Play with Half-Quadratic Splitting.

Extension points:
    - apply_forward / apply_adjoint
    - solve_hqs(H, g, z, alpha, params) : HQS inversion step

5. BaseMcmc
------------------------------------------------
MCMC MMSE estimator via Metropolis-Hastings.

Extension points:
    - apply_forward / apply_adjoint
    - proposal_step(f, g, H, sigma, denoiser, params)

---------------------------------------------------------------------
UI parameters (ui_params)
---------------------------------------------------------------------

Each algorithm carries a ui_params dictionary that defines its
configurable parameters for the GUI. These dictionaries live
alongside their algorithm (e.g. adam/ui_params.py).

The ui_params support:
    - requires={...}  : parameter shown only if features match
    - depends_on={...}: parameter shown only if another option matches

This mechanism is automatic: when matirf sets features={"3d"},
parameters requiring {"3d"} appear; when deconv sets features={"2d"},
they are filtered out. No code needed in the problem-specific modules.

---------------------------------------------------------------------
Design philosophy
---------------------------------------------------------------------

This layer defines the "algorithm architecture contract":

    base          -> algorithmic primitives
    reusable      -> ready-to-use regularizations and data fidelities
    specializable -> full algorithm skeletons with hooks

Key principles:
    - enforce consistent algorithm structure across all inverse problems
    - centralize iterative logic (convergence, scheduling, logging)
    - isolate problem-specific logic into small override hooks
    - reduce duplication of algorithm implementations

---------------------------------------------------------------------
Summary
---------------------------------------------------------------------

SPECIALIZABLE = fully functional algorithms with extension points.

They define *how the algorithm iterates*, but not *what the operator is*.
"""

from .adam import BaseAdam, ADAM_UI_PARAMETERS
from .admm import BaseAdmm, ADMM_UI_PARAMETERS
from .mcmc import BaseMcmc, MCMC_UI_PARAMETERS
from .pnp import BasePnp, PNP_UI_PARAMETERS
from .ppxa import BasePpxa, PPXA_UI_PARAMETERS
