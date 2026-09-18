from common.gui.specializable.display_window import BaseDisplayWindow
from .display_window_manager import DeconvDisplayWindowManager
from .figures_section import DeconvFiguresSection
from .synthetic_truth_section import DeconvSyntheticTruthSection
from deconv.core import DeconvPipeline
from deconv import DECONV_RESULTS_DIR


class DeconvDisplayWindow(BaseDisplayWindow):
    """Deconvolution display window — configured entirely by the class attributes below."""

    WINDOW_TITLE = "Deconv Display Window"
    RESULTS_DIR = DECONV_RESULTS_DIR
    PIPELINE_CLASS = DeconvPipeline
    DISPLAY_WINDOW_MANAGER_CLASS = DeconvDisplayWindowManager
    FIGURES_SECTION_CLASS = DeconvFiguresSection
    SYNTHETIC_SECTION_CLASS = DeconvSyntheticTruthSection
