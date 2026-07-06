from common.gui.specializable.control_window import BaseControlWindow
from common.gui.reusable.algorithm_selection_section import AlgorithmSelectionSection
from common.gui.reusable.add_noise_section import ADD_NOISE_PARAMETERS_UI
from .sections import InputFilesSection
from .sections.operator_parameters_section.operator_parameters_ui_dictionary import OPERATOR_PARAMETERS_UI
from .display_window_manager import DisplayWindowManager
from matirf import MATIRF_CONFIG_PATH, MATIRF_RESULTS_DIR, DEFAULT_MATIRF_CONFIG, MATIRF_FEATURES
from matirf.core import MaTirfPipeline
from matirf.algorithms import ALGORITHMS

class ControlWindow(BaseControlWindow):
    """MA-TIRF control window.

    The window is built entirely from the class attributes below.
    BaseControlWindow reads them to create the UI, build the sections,
    and wire up config load/save/run.

    The only method to override is on_close_cleanup(), which closes
    sub-windows (editors, viewers) that are specific to this problem.
    """

    window_title = "MA-TIRF Parameter Selection"
    cached_config_path = MATIRF_CONFIG_PATH
    default_config = DEFAULT_MATIRF_CONFIG
    results_dir = MATIRF_RESULTS_DIR
    pipeline_class = MaTirfPipeline
    display_window_manager_class = DisplayWindowManager

    # ── section descriptors ─────────────────────────────────────────────

    sections_left = [
        InputFilesSection,
        ("Operator Parameters", OPERATOR_PARAMETERS_UI, 'oper-params'),
        ("Add noise to measurement", ADD_NOISE_PARAMETERS_UI, 'add-noise'),
    ]

    sections_right = [
        # The "Estimate" button for delta is defined directly in the algorithm
        # UI dicts (e.g. ADAM_UI_PARAMETERS['delta']['extra_button']).
        (AlgorithmSelectionSection, {'algorithms_dict': ALGORITHMS, 'problem_features': MATIRF_FEATURES}),
    ]

    def on_close_cleanup(self):
        """Close sub-windows that are specific to MA-TIRF."""
        if self.input_files_section.json_selector.measurement_parameters_editor:
            self.input_files_section.json_selector.measurement_parameters_editor.close()
        if self.input_files_section.image_selector.tif_file_preprocess_editor:
            self.input_files_section.image_selector.tif_file_preprocess_editor.close()
