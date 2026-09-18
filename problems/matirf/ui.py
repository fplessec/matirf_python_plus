"""
The MA-TIRF interface, declared.

This one file replaces the nine that used to live under `problems/matirf/gui/`: a control
window, a display window, a window manager, a figures section, a synthetic-truth section,
an input-files section and two "Estimate" callbacks. `gui/factory.py` builds all of them
from the declaration at the bottom.

Read it top to bottom: the small helpers first (what a preview shows, how a depth view is
refreshed, what each Estimate button computes), then MATIRF_UI, which assembles them.
"""

import torch

from gui.reusable import SingularValuePickerDialog
from gui.spec import Editor, Estimator, FileSlot, Panel, Preview, ProblemUI, View, image_panel
from gui.widgets import DepthMapViewer, ImageAndHisto3DViewer, ProfilesViewer
from gui.widgets.figure_export import export_depth_and_profiles

from .parameters import MEASUREMENT_PARAMETERS_UI
from .operator_parameters import OPERATOR_PARAMETERS_UI


# ── the preprocessing preview ─────────────────────────────────────────────────

PREVIEW_TITLES = {
    "real": ("g_raw = input file (MA-TIRF image stack, raw measurement)",
             "g = preprocessed g_raw + noise"),
    "synthetic": ("f_true = input file (3D object, synthetic truth)",
                  "g_synth = H_synth @ f_true (preprocessed + noise)"),
}


def _preview(config, mode):
    """
    The problem previews itself, through the SAME code path a run takes — so what the
    preview shows can never disagree with the reconstruction that follows.
    """
    from problems.matirf import MATIRF
    return MATIRF.preview(config)


def _preview_errors(config):
    """Readable reasons the preview could not be computed, shown instead of a traceback."""
    add_noise, oper = config.get("add-noise", {}), config.get("oper-params", {})
    messages = []
    if add_noise.get("add_noise", False) and add_noise.get("sigma", "None") == "None":
        messages.append("Noise standard deviation is None, please define a value.")
    for key, label in (("nz", "Number of cuts on z"), ("z0", "Smallest depth"),
                       ("zN", "Largest depth")):
        if oper.get(key, "None") == "None":
            messages.append(f"{label} is None, please define a value.")
    ## a checkbox never touched is absent from the TOML; False is a valid answer, not a gap
    if oper.get("normalize", "None") in (None, "None", "null"):
        messages.append("Normalize Operator is not set, please tick or untick the box.")
    return messages


# ── the figure views ──────────────────────────────────────────────────────────
# View 1 needs z0 / zN to label its depth axis, so its panels read them from the config
# rather than using the generic image_panel helper.

def _z_bounds(config):
    return config["oper-params"]["z0"], config["oper-params"]["zN"]


def _build_depth_map(f, config):
    z0, zN = _z_bounds(config)
    return DepthMapViewer(f, z0, zN)


def _build_profiles(f, config):
    z0, zN = _z_bounds(config)
    return ProfilesViewer(f, z0, zN)


def _refresh_z_view(widget, f, config):
    """Re-apply the z bounds, then the image: the depth axis may have changed too."""
    widget.z0, widget.zN = _z_bounds(config)
    widget.set_image(f)


def _export_view1(view_widgets, filepath):
    """view_widgets is this view's panels, in order: [depth map, profiles]."""
    export_depth_and_profiles(view_widgets[0], view_widgets[1], filepath)


# ── the Estimate buttons ──────────────────────────────────────────────────────
# Both are one line, because both quantities are properties of the PHYSICS and therefore
# live on the operator. The validation, the error reporting, the writing and the widget
# refresh are handled once in gui/estimators.py.

ESTIMATE_DELTA = Estimator(
    compute=lambda operator, config: operator.estimate_anisotropy_ratio(),
    tooltip="Estimate delta from the measurement parameters (.json) "
            "and the operator parameters (nz, z0, zN).",
)

ESTIMATE_LAMBDA_RR = Estimator(
    compute=lambda operator, config: torch.linalg.svdvals(operator.H),
    tooltip="Estimate lambda_rr from the singular value spectrum of H.\n"
            "Computes the SVD of the MA-TIRF operator and lets you\n"
            "choose which singular value to use as cutoff.",
    picker=SingularValuePickerDialog,
    title="Estimate lambda_rr from the SVD of H",
    info=lambda operator, config: (
        f"H shape: {tuple(operator.H.shape)}    "
        f"cond(H): {_condition_number(operator.H):.1f}"),
    picker_kwargs={"spectrum_label_prefix": "s"},
)


def _condition_number(H) -> float:
    s = torch.linalg.svdvals(H)
    return float(s[0] / s[-1])


# ── the declaration ───────────────────────────────────────────────────────────

MATIRF_UI = ProblemUI(
    control_title="MA-TIRF Parameter Selection",
    display_title="Display Window",
    real_mode_text="Work with real MA-TIRF measurement",
    synthetic_mode_text="Simulate measurement with synthetic truth",

    image_slot=FileSlot(
        toml_key="tif", noun="tif", dialog_filter="Image Files (*.tif *.tiff)",
        title_real="Path of the MA-TIRF image stack",
        title_synthetic="Path of the 3D object (synthetic truth)",
        preview=Preview(ImageAndHisto3DViewer, _preview, titles=PREVIEW_TITLES,
                        size=(700, 900), error_check=_preview_errors),
    ),
    json_slot=FileSlot(
        toml_key="json", noun="json", dialog_filter="Parameters Files (*.json)",
        title_real="Path of the measurement parameters",
        title_synthetic="Path of the simulated parameters",
        require_keys=MEASUREMENT_PARAMETERS_UI,          # the .json must carry these
        editor=Editor(ui=MEASUREMENT_PARAMETERS_UI,
                      title_noun="Measurement Parameters", width=700),
    ),

    parameters=[("Operator Parameters", OPERATOR_PARAMETERS_UI, "oper-params")],

    views=[
        View("view 1",
             [Panel(_build_depth_map, _refresh_z_view, stretch=2),
              Panel(_build_profiles, _refresh_z_view, stretch=1)],
             export=_export_view1),
        View("view 2", [image_panel(ImageAndHisto3DViewer, title="")]),
    ],

    difference_viewer=ImageAndHisto3DViewer,
    estimators={"delta": ESTIMATE_DELTA, "lambda_rr": ESTIMATE_LAMBDA_RR},
)
