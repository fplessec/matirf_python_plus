"""
SPECIALIZABLE ALGORITHM LAYER (Extensible algorithm framework)

This module defines partially implemented iterative algorithms that form
the computational backbone of inverse problem solvers (matirf, deconv,
and future extensions).

Unlike the BASE layer (algorithm machinery) and the REUSABLE layer
(optimization terms), this layer defines full algorithm skeletons with
explicit extension points ("hooks") that must be implemented by
problem-specific modules.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------
Each algorithm is functional but not fully defined:
    > it implements the full iterative loop and convergence logic
    > it exposes a small set of hooks for customization
    > it enforces architectural consistency across problems
    > it carries its own ui_params dictionary for GUI integration

They define how the algorithm iterates, but not what the operator is.

---------------------------------------------------------------------
The forward-model hooks (shared by all algorithms)
---------------------------------------------------------------------
Every algorithm inherits two mandatory hooks from Algorithm (base/),
which each inverse problem must implement:
    > apply_forward(H, f)  -> computes Hf
    > apply_adjoint(H, g)  -> computes H^t g

These are the single plug point for the forward model. The per-algorithm
hooks listed below come in addition to them.

---------------------------------------------------------------------
Main components
---------------------------------------------------------------------
> BaseAdam
    > Gradient-based optimizer using torch.optim.Adam.
    > Extra hooks: init_f(g, H, params)  (optional, default H^t g)

> BasePpxa
    > Parallel Proximal Algorithm (PPXA).
    > Extra hooks:
        - init_f, precompute
        - compute_data_prox(u, params)   proximal operator of the data term

> BaseAdmm
    > Alternating Direction Method of Multipliers.
    > Extra hooks:
        - init_f, precompute
        - solve_linear_system(Htg, v, mu, params)
        - compute_threshold(g, H, params)

> BasePnp
    > Plug-and-Play with Half-Quadratic Splitting.
    > Extra hooks: solve_hqs(H, g, z, alpha, params)   HQS inversion step

> BaseMcmc
    > MCMC MMSE estimator via Metropolis-Hastings.
    > Extra hooks:
        - init_f
        - proposal_step(f, g, H, sigma, denoiser, params)

---------------------------------------------------------------------
UI parameters (ui_params)
---------------------------------------------------------------------
Each algorithm carries an ui_params dictionary defining its configurable
parameters for the GUI. These dictionaries live alongside their
algorithm (e.g. adam/ui_params.py).

They support conditional visibility:
    > requires={...}   -> parameter shown only if the problem features match
    > depends_on={...} -> parameter shown only if another option matches

This is automatic: when matirf sets features={"3d"}, parameters requiring
{"3d"} appear; when deconv sets features={"2d"}, they are filtered out.
No code is needed in the problem-specific modules.

---------------------------------------------------------------------
---------------------------------------------------------------------
Overall design philosophy of Specializable terms:
    > enforce a consistent algorithm structure across all inverse problems
    > centralize iterative logic (convergence, scheduling, logging)
    > isolate problem-specific logic into small override hooks
    > reduce duplication of algorithm implementations
"""

from .adam import BaseAdam
from .admm import BaseAdmm
from .mcmc import BaseMcmc
from .pnp import BasePnp, BasePnpAdmm
from .ppxa import BasePpxa
