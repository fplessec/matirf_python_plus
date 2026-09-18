from common.gui.specializable.display_window import BaseDisplayWindow
from .figures_section import FiguresSection
from .synthetic_truth_section import SyntheticTruthSection
from .display_window_manager import DisplayWindowManager
from pipeline import pipeline_for
from problems.matirf import MATIRF
from matirf import MATIRF_RESULTS_DIR


class DisplayWindow(BaseDisplayWindow):
    """MA-TIRF display window — configured entirely by the class attributes below."""

    WINDOW_TITLE = "Display Window"
    RESULTS_DIR = MATIRF_RESULTS_DIR
    PIPELINE_CLASS = pipeline_for(MATIRF)
    DISPLAY_WINDOW_MANAGER_CLASS = DisplayWindowManager
    FIGURES_SECTION_CLASS = FiguresSection
    SYNTHETIC_SECTION_CLASS = SyntheticTruthSection
