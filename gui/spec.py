"""
ProblemUI — one inverse problem's entire interface, declared.

--------------------------------------------------------------------------------------
What this replaces
--------------------------------------------------------------------------------------

Before this file, every problem carried nine interface files: a control window, a display
window, a window manager, a figures section, a synthetic-truth section, an input-files
section, two parameter dictionaries and two "Estimate" callbacks. Each was short and
declarative — but there were nine of them, in two parallel trees, and adding a problem
meant writing all nine again.

They were nine because each described a DIFFERENT PART of the same interface. Gather those
parts into one object and the nine collapse into one: `gui/factory.py` builds both windows
from a `ProblemUI`, so a problem writes a declaration and no window code at all.

--------------------------------------------------------------------------------------
The shape of a declaration
--------------------------------------------------------------------------------------

    MATIRF_UI = ProblemUI(
        control_title = "MA-TIRF Parameter Selection",
        display_title = "Display Window",
        real_mode_text      = "Work with real MA-TIRF measurement",
        synthetic_mode_text = "Simulate measurement with synthetic truth",
        image_slot = FileSlot(...),          # the measurement file
        json_slot  = FileSlot(...),          # its parameters
        parameters = [("Operator Parameters", OPERATOR_UI, "oper-params"),
                      ("Add noise",           ADD_NOISE_PARAMETERS_UI, "add-noise")],
        views      = [View("view 1", [...]), View("view 2", [...])],
        difference_viewer = ImageAndHisto3DViewer,
        estimators = {"delta": Estimator(...), "lambda_rr": Estimator(...)},
    )

Nothing here mentions Qt. A declaration says WHAT the interface offers; `gui/` decides how
it is drawn, which is what lets the drawing change once for every problem at a time.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

## the per-part specs already used by the generic sections — re-exported so a problem's
## ui.py imports everything it needs from this one module:
from gui.specializable.control_window.sections.base_input_files_section import (  # noqa: F401
    FileSlot, Preview, Editor,
)
from gui.specializable.display_window.sections.base_figures_section import (  # noqa: F401
    View, Panel, image_panel,
)


@dataclass(frozen=True)
class Estimator:
    """
    An "Estimate" button on a parameter the user would otherwise have to compute by hand.

    Both of this project's estimators — MA-TIRF's anisotropy ratio and the ridge weight —
    had the same shape written out twice, by hand: check that everything needed is set,
    report what is missing as a bulleted list, build the operator, compute, catch failures,
    write the result, refresh the widget. Only the computation and the dialog differed.

    Everything except the computation is now done once, in `gui/estimators.py`. In
    particular the precondition check is `problem.validate(config)` — the problem already
    knows what a usable configuration is, so the estimator does not restate it.

    ----------
    > Fields :
    ----------

    >> compute : callable(operator, config)
        The computation, given an already-built operator. Returns a float for a direct
        estimate, or the values to display when `picker` is set. It receives the operator
        rather than the config alone, because the quantity being estimated is a property of
        the physics — which is why these computations live on the operator, not here.

    >> tooltip : str
        What the button explains when hovered. Worth writing: it is where a user learns
        what the estimate is based on.

    >> label : str
        The button's text. "Estimate" unless there is a reason.

    >> picker : type or None
        An optional dialog class for an estimate the user must CHOOSE rather than accept —
        picking a cutoff in a spectrum, for instance. When set, `compute` returns the values
        to plot and the dialog returns the selection through `selected_lambda_rr()`.

    >> title / info : str / callable(operator, config) -> str
        The dialog's title, and a one-line subtitle computed from the operator (shapes,
        conditioning) to help the user choose. Ignored without a `picker`.

    >> picker_kwargs : dict
        Extra keyword arguments the dialog class needs, e.g. a label prefix.

    ----------
    > Example :
    ----------

        "delta": Estimator(
            compute=lambda operator, config: operator.estimate_anisotropy_ratio(),
            tooltip="Estimate delta from the measurement parameters and nz / z0 / zN.",
        )

        "lambda_rr": Estimator(
            compute=lambda operator, config: torch.linalg.svdvals(operator.H),
            picker=SingularValuePickerDialog,
            title="Estimate lambda_rr from the SVD of H",
            info=lambda operator, config: f"H shape: {tuple(operator.H.shape)}",
            picker_kwargs={"spectrum_label_prefix": "s"},
        )
    """

    compute: Callable
    tooltip: str = ""
    label: str = "Estimate"
    picker: Optional[type] = None
    title: str = ""
    info: Optional[Callable] = None
    picker_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemUI:
    """
    Everything the interface of one inverse problem needs to know.

    ----------
    > Fields :
    ----------

    >> control_title / display_title : str
        The titles of the two windows.

    >> real_mode_text / synthetic_mode_text : str
        The labels either side of the real/synthetic toggle. They are worth phrasing in the
        problem's own words ("Simulate measurement with synthetic truth"), because that
        toggle is the single most consequential choice in the window.

    >> image_slot / json_slot : FileSlot
        The two input files: the measurement (or the ground truth, in synthetic mode) and
        its parameters. A FileSlot may carry a `Preview` — the "see preprocessed file"
        popup — and an `Editor` for the .json.

    >> parameters : list[(title, ui_dict, toml_key)]
        The parameter sections of the control window, in order, top to bottom. Each is a
        plain params-UI dictionary bound to a section of config.toml. `("Add noise",
        ADD_NOISE_PARAMETERS_UI, "add-noise")` is shared by every problem.

    >> views : list[View]
        The switchable views of the display window. Each View holds Panels — one viewer
        widget each. `image_panel(SomeViewer)` covers the common case.

    >> difference_viewer : type or None
        The viewer used for the f_true - alpha*f popup in synthetic mode. None means the
        problem has no synthetic mode.

    >> estimators : dict[str, Estimator]
        "Estimate" buttons, keyed by the algorithm parameter they fill in. A key absent from
        a given solver is simply skipped, so one mapping serves the whole solver registry.

    >> export_default_dir : Path or str
        Where the "Save view as PNG" dialog opens. Defaults to the problem's results
        directory.
    """

    control_title: str
    display_title: str = "Display Window"
    real_mode_text: str = "Work with real measurement"
    synthetic_mode_text: str = "Simulate with synthetic truth"
    image_slot: Optional[FileSlot] = None
    json_slot: Optional[FileSlot] = None
    parameters: tuple = ()
    views: tuple = ()
    difference_viewer: Optional[type] = None
    estimators: dict = field(default_factory=dict)
    export_default_dir: Optional[object] = None

    @property
    def supports_synthetic_view(self) -> bool:
        return self.difference_viewer is not None
