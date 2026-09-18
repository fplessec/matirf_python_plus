"""
Running an "Estimate" button — the part that was written out once per estimator.

Both of this project's estimators followed the same six steps, copied by hand:

    1. read the cached config
    2. check everything needed is set, and list what is missing
    3. show that list as a warning, or carry on
    4. build the operator and compute
    5. report a failure as a warning rather than a traceback
    6. write the result into the config and refresh the widget

Only step 4 differed. The rest is here, once, and an `Estimator` supplies step 4.

Step 2 is the interesting simplification: instead of each estimator restating which keys it
needs — which is how the two v1 callbacks came to carry thirty lines of duplicated
validation between them — it calls `problem.validate(config)`. The problem already knows
what a usable configuration is, and an estimate needs exactly the same things a run does:
the files, and the parameters the operator is built from.

The GUI is allowed to know about the problem; the problem never knows about the GUI. That
direction is what keeps the algorithm layer free of the circular import v1 suffered from.
"""

from PyQt5.QtWidgets import QMessageBox


def attach(solvers: dict, estimators: dict, problem) -> dict:
    """
    A view of the solver registry whose parameters carry this problem's Estimate buttons.

    Returns fresh subclasses; the shared solver classes are never modified, so one problem's
    buttons can never leak into another's window — the failure mode of editing a class-level
    ui_params dictionary in place.
    """
    from gui.reusable import with_extra_buttons

    buttons = {
        key: {"label": spec.label,
              "tooltip": spec.tooltip,
              "callback": _callback_for(key, spec, problem)}
        for key, spec in (estimators or {}).items()
    }
    return with_extra_buttons(solvers, buttons) if buttons else dict(solvers)


def _callback_for(param_key: str, spec, problem):
    """Build the callback SimpleParameterWidget expects for an extra button."""

    def run(widget, update_cache_fn, load_toml_fn, config_path):
        config = load_toml_fn(config_path)

        # 1. the problem's own validation IS the precondition for estimating
        problems = problem.validate(config)
        if problems:
            _warn(widget, param_key,
                  "The following must be set first:\n\n"
                  + "\n".join(f"  • {p}" for p in problems))
            return

        # 2. build the operator and compute
        try:
            operator = problem.make_operator(config)
            result = spec.compute(operator, config)
        except Exception as error:
            _warn(widget, param_key, f"{type(error).__name__}: {error}")
            return

        def write(value):
            update_cache_fn(["algo-params", param_key], float(value))
            widget.update_ui_from_toml(config_path)

        # 3a. a direct estimate: write it
        if spec.picker is None:
            write(result)
            return

        # 3b. an estimate the user must choose: show the values and write the selection.
        # The dialog is modeless (show, not exec_) so the window stays usable while the
        # user compares; the write happens on its accepted signal.
        info = spec.info(operator, config) if spec.info else ""
        dialog = spec.picker(widget, title=spec.title or f"Estimate {param_key}",
                             info_text=info, S=result, **spec.picker_kwargs)
        dialog.accepted.connect(lambda: write(dialog.selected_lambda_rr()))
        dialog.show()

    return run


def _warn(widget, param_key: str, message: str) -> None:
    QMessageBox.warning(widget, f"Cannot estimate {param_key}", message)
