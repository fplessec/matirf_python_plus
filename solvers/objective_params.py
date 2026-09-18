"""
The parameters that describe the OBJECTIVE, shared by every solver that uses one.

`data_fidelity`, `reg`, `lambda_reg`, `delta`, `rho` do not belong to any particular
algorithm — they describe what is being minimized, not how. v1 nevertheless repeated them
inside every algorithm's ui_params (adam, ppxa, admm, pnp, mcmc), so adding a new
regularization meant editing five files and forgetting one was easy.

Here they are declared once. Any solver with `uses_regularization = True` receives them
automatically (see `Solver.get_ui_params`), and its own ui_params contain only what is
genuinely specific to that algorithm — iterations, learning rate, step size.

The fidelities and regularizations themselves live under `solvers/` — they describe the
objective, not any problem, so this is where they belong. The old paths under
`common/algorithms/reusable/` are re-export shims kept only until the v1 stack is deleted.
"""

from solvers.fidelities import DATA_FIDELITY_LIST
from solvers.regularizers import (
    REGULARIZATION_LIST, ANISOTROPIC_REGULARIZATIONS,
)
from core.features import Feature


## Which noise model D(Hf, g) measures the discrepancy. Offered to any solver that
## evaluates D through the objective (Adam, PPXA, MCMC) — but NOT to the splitting solvers
## whose data step is a hardcoded quadratic (ADMM, PnP, PnP-ADMM), where choosing a Poisson
## fidelity would have no effect and would only mislead the user.
DATA_FIDELITY_UI_PARAM = {
    "data_fidelity": {
        "title": "Data fidelity (noise model)",
        "type": "option",
        "param_info": {"options_list": DATA_FIDELITY_LIST},
    },
}


## The explicit prior R(f) and its weight. Offered only to solvers that actually consult
## the objective's regularization term.
REGULARIZATION_UI_PARAMS = {
    "reg": {
        "title": "Regularization",
        "type": "option",
        "param_info": {"options_list": REGULARIZATION_LIST},
    },
    "lambda_reg": {
        "title": "Regularization coefficient",
        "type": "value",
        "param_info": {
            "dtype": float,
            "unit": "",
            "latex_name": "\\lambda_{reg}",
            "default": 0.10,
        },
    },
    # shown only on anisotropic problems, and only for regularizations that weight the
    # axial derivative — the framework hides it everywhere else with no code in the solver:
    "delta": {
        "title": "Anisotropy ratio coefficient",
        "type": "value",
        "requires": {Feature.ANISOTROPIC},
        "depends_on": {"reg": ANISOTROPIC_REGULARIZATIONS},
        "param_info": {
            "dtype": float,
            "unit": "",
            "latex_name": "\\delta = \\frac{\\Delta z}{\\Delta xy}",
            "default": 0.05,
        },
    },
    "rho": {
        "title": "Sparsity coefficient for SHV",
        "type": "value",
        "depends_on": {"reg": "sparse hessian variation"},
        "param_info": {
            "dtype": float,
            "unit": "",
            "latex_name": "\\rho",
            "default": 0.6,
        },
    },
}


## Everything describing the objective — what `Solver.get_ui_params` merges in.
OBJECTIVE_UI_PARAMS = {**DATA_FIDELITY_UI_PARAM, **REGULARIZATION_UI_PARAMS}


## Where the iteration starts. Generic since ForwardOperator.ridge_inverse exists for every
## problem — in v1 the ridge start was a MA-TIRF-only subclass override.
INIT_UI_PARAM = {
    "init": {
        "title": "Initialization",
        "type": "option",
        "param_info": {"options_list": ["adjoint", "ridge"]},
    },
    "lambda_rr": {
        "title": "Ridge regularization (initialization)",
        "type": "value",
        "depends_on": {"init": "ridge"},
        "param_info": {
            "dtype": float,
            "unit": "",
            "latex_name": "\\lambda_{rr}",
            "default": 10000.0,
        },
    },
}
