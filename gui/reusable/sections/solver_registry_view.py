"""
Presenting the solver registry to one problem's GUI, with its own extra buttons.

A solver's `ui_params` is deliberately pure data — no callbacks, no imports from the GUI.
That is what keeps the algorithm layer free of the circular dependency v1 suffered from.
But some parameters are genuinely easier to compute than to type: MA-TIRF's anisotropy
ratio `delta` follows from the measurement, and the ridge weight `lambda_rr` follows from
the operator's spectrum. Both deserve an "Estimate" button.

The button therefore belongs to the GUI, which is allowed to know about the problem. This
module builds a VIEW of the registry — a set of thin subclasses whose `get_ui_params`
returns the same dictionary with the buttons attached.

Why a subclass rather than editing the dictionary in place: `ui_params` is a class
attribute shared by every window and every problem. Mutating it would leak MA-TIRF's
buttons into deconvolution, and a second call would stack duplicates. That exact bug
(a shared UI dict mutated by an editor) has already been fixed once in this project;
this avoids repeating it.
"""

import copy


def with_extra_buttons(solvers: dict, buttons: dict, defaults: dict = None) -> dict:
    """
    A copy of the solver registry whose parameters carry problem-specific extra buttons.

    ----------
    > Parameters :
    ----------

    >> solvers : dict
        The registry to present, normally `solvers.SOLVERS` (name -> Solver subclass).

    >> buttons : dict
        Which button to attach to which parameter, as {param_key: button_spec}, where
        button_spec is the `extra_button` dict SimpleParameterWidget understands:

            {"label": "Estimate",
             "tooltip": "...",
             "callback": fn(widget, update_cache_fn, load_toml_fn, config_path)}

        A parameter absent from a given solver is simply skipped, so one mapping can be
        written per problem and applied to the whole registry.

    >> defaults : dict, optional
        The problem's `solver_defaults`, {solver name: {param: value}}. The view shows them
        AS the defaults — an option moved first, a value's default replaced — so what the
        window displays for an untouched field is what the pipeline will use.

    ----------
    > Returns :
    ----------
        A new dict of the same shape. The original classes are untouched.

    ----------
    > Example :
    ----------

        SOLVERS_FOR_MATIRF = with_extra_buttons(SOLVERS, {
            "delta":     {"label": "Estimate", "tooltip": "...", "callback": estimate_delta},
            "lambda_rr": {"label": "Estimate", "tooltip": "...", "callback": estimate_lambda_rr},
        })
    """
    defaults = defaults or {}
    return {name: _view_of(solver, buttons, defaults.get(name, {}))
            for name, solver in solvers.items()}


def _view_of(solver, buttons: dict, defaults: dict) -> type:
    """A subclass of `solver` whose get_ui_params adds the buttons, leaving the original alone."""

    @classmethod
    def get_ui_params(cls, problem_features=None) -> dict:
        params = copy.deepcopy(solver.get_ui_params(problem_features))
        for key, button in buttons.items():
            if key in params:
                params[key] = {**params[key], "extra_button": button}
        for key, value in defaults.items():
            info = params.get(key, {}).get("param_info")
            if info is None:
                continue
            if "options_list" in info and value in info["options_list"]:
                info["options_list"] = [value] + [o for o in info["options_list"] if o != value]
            elif "default" in info:
                info["default"] = value
        return params

    return type(solver.__name__, (solver,), {"get_ui_params": get_ui_params})
