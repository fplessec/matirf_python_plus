from common.gui.specializable.control_window import BaseControlWindow
from common.gui.reusable import (
    AlgorithmSelectionSection, ADD_NOISE_PARAMETERS_UI, with_extra_buttons,
)
from solvers import SOLVERS
from pipeline import pipeline_for
from problems.deconv import DECONV
from deconv import DECONV_CONFIG_PATH, DECONV_RESULTS_DIR, DEFAULT_DECONV_CONFIG
from .input_files_section import DeconvInputFilesSection
from .display_window_manager import DeconvDisplayWindowManager
from .estimate_lambda_rr import estimate_lambda_rr


## Same declarative pattern as MA-TIRF, with one estimate button instead of two: there is no
## `delta` on an isotropic 2D problem, so the framework never shows that parameter and the
## button mapping for it would simply be skipped. See matirf/gui/control_window.py.
DECONV_SOLVERS = with_extra_buttons(SOLVERS, {
    "lambda_rr": {
        "label": "Estimate",
        "tooltip": "Estimate lambda_rr from the Fourier spectrum of the PSF.\n"
                   "Displays the sorted |H_fft| values and lets you\n"
                   "choose which frequency to use as cutoff.",
        "callback": estimate_lambda_rr,
    },
})


class DeconvControlWindow(BaseControlWindow):
    """Deconvolution control window.

    Same declarative pattern as the MA-TIRF ControlWindow; see that file for the
    detailed comments on the architecture.
    """

    window_title = "Deconvolution Parameter Selection"
    cached_config_path = DECONV_CONFIG_PATH
    default_config = DEFAULT_DECONV_CONFIG
    results_dir = DECONV_RESULTS_DIR
    pipeline_class = pipeline_for(DECONV)
    display_window_manager_class = DeconvDisplayWindowManager

    # ── section descriptors ─────────────────────────────────────────────

    sections_left = [
        DeconvInputFilesSection,
        ("Add noise to measurement", ADD_NOISE_PARAMETERS_UI, 'add-noise'),
    ]

    sections_right = [
        (AlgorithmSelectionSection,
         {'algorithms_dict': DECONV_SOLVERS, 'problem_features': DECONV.features}),
    ]

    def on_close_cleanup(self):
        """Close sub-windows (editor / preview) opened from the input-files selectors."""
        self.deconv_input_files_section.json_selector.close_sub_window()
        self.deconv_input_files_section.image_selector.close_sub_window()
