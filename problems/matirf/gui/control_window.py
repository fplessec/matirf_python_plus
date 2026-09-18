from gui.specializable.control_window import BaseControlWindow
from gui.reusable import (
    AlgorithmSelectionSection, ADD_NOISE_PARAMETERS_UI, with_extra_buttons,
)
from solvers import SOLVERS
from pipeline import pipeline_for
from problems.matirf import MATIRF
from problems.matirf import MATIRF_CONFIG_PATH, MATIRF_RESULTS_DIR, DEFAULT_MATIRF_CONFIG
from .input_files_section import InputFilesSection
from .operator_parameters_ui_dictionary import OPERATOR_PARAMETERS_UI
from .display_window_manager import DisplayWindowManager
from .estimate_delta import estimate_delta
from .estimate_lambda_rr import estimate_lambda_rr


## The solvers are shared and problem-agnostic, so their ui_params carry no callbacks — that
## is what keeps the algorithm layer free of any GUI import. The two parameters below are
## easier to compute than to type, so the GUI attaches an "Estimate" button to them here,
## where knowing about MA-TIRF is allowed. `with_extra_buttons` returns a VIEW: the shared
## solver classes are never modified, so deconvolution does not inherit these buttons.
MATIRF_SOLVERS = with_extra_buttons(SOLVERS, {
    "delta": {
        "label": "Estimate",
        "tooltip": "Estimate delta from the measurement parameters (.json) "
                   "and the operator parameters (nz, z0, zN).",
        "callback": estimate_delta,
    },
    "lambda_rr": {
        "label": "Estimate",
        "tooltip": "Estimate lambda_rr from the singular value spectrum of H.\n"
                   "Computes the SVD of the MA-TIRF operator and lets you\n"
                   "choose which singular value to use as cutoff.",
        "callback": estimate_lambda_rr,
    },
})


class ControlWindow(BaseControlWindow):
    """MA-TIRF control window.

    Built entirely from the class attributes below: BaseControlWindow reads them to
    create the UI, build the sections, and wire config load/save/run.

    The only method to override is on_close_cleanup(), which closes the sub-windows
    (editors, viewers) this problem's input section can open.
    """

    window_title = "MA-TIRF Parameter Selection"
    cached_config_path = MATIRF_CONFIG_PATH
    default_config = DEFAULT_MATIRF_CONFIG
    results_dir = MATIRF_RESULTS_DIR
    pipeline_class = pipeline_for(MATIRF)
    display_window_manager_class = DisplayWindowManager

    # ── section descriptors ─────────────────────────────────────────────

    sections_left = [
        InputFilesSection,
        ("Operator Parameters", OPERATOR_PARAMETERS_UI, 'oper-params'),
        ("Add noise to measurement", ADD_NOISE_PARAMETERS_UI, 'add-noise'),
    ]

    sections_right = [
        (AlgorithmSelectionSection,
         {'algorithms_dict': MATIRF_SOLVERS, 'problem_features': MATIRF.features}),
    ]

    def on_close_cleanup(self):
        """Close sub-windows (editor / preview) opened from the input-files selectors."""
        self.input_files_section.json_selector.close_sub_window()
        self.input_files_section.image_selector.close_sub_window()
