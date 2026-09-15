from common.gui.specializable.control_window import BaseControlWindow
from common.gui.reusable import AlgorithmSelectionSection, ADD_NOISE_PARAMETERS_UI
from .sections import DeconvInputFilesSection
from .display_window_manager import DeconvDisplayWindowManager
from deconv import DECONV_CONFIG_PATH, DECONV_RESULTS_DIR, DEFAULT_DECONV_CONFIG, DECONV_FEATURES
from deconv.core import DeconvPipeline
from deconv.algorithms import DECONV_ALGORITHMS


class DeconvControlWindow(BaseControlWindow):
    """Deconvolution control window.

    Same declarative pattern as the MA-TIRF ControlWindow.
    See BaseControlWindow and matirf/gui/control_window/control_window.py
    for detailed comments on the architecture.
    """

    window_title = "Deconvolution Parameter Selection"
    cached_config_path = DECONV_CONFIG_PATH
    default_config = DEFAULT_DECONV_CONFIG
    results_dir = DECONV_RESULTS_DIR
    pipeline_class = DeconvPipeline
    display_window_manager_class = DeconvDisplayWindowManager

    # ── section descriptors ─────────────────────────────────────────────

    sections_left = [
        DeconvInputFilesSection,
        ("Add noise to measurement", ADD_NOISE_PARAMETERS_UI, 'add-noise'),
    ]

    sections_right = [
        # No extra_widget_factory needed for deconv (no "Estimate" button)
        (AlgorithmSelectionSection, {'algorithms_dict': DECONV_ALGORITHMS, 'problem_features': DECONV_FEATURES}),
    ]

    # ── auto-named attributes ───────────────────────────────────────────
    # DeconvInputFilesSection    → self.deconv_input_files_section
    # 'add-noise'                → self.add_noise
    # AlgorithmSelectionSection  → self.algorithm_selection_section

    def on_close_cleanup(self):
        """Close sub-windows (editor / preview) opened from the input-files selectors."""
        self.deconv_input_files_section.json_selector.close_sub_window()
        self.deconv_input_files_section.image_selector.close_sub_window()
