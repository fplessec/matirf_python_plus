from common.gui.specializable.display_window import BaseDisplayWindow
from .sections import FiguresSection, SyntheticTruthSection
from matirf.gui.control_window import DisplayWindowManager
from matirf.core import MaTirfPipeline
from matirf import MATIRF_RESULTS_DIR


class DisplayWindow(BaseDisplayWindow):

    def window_title(self):
        return "Display Window"

    def results_dir(self):
        return MATIRF_RESULTS_DIR

    def pipeline_class(self):
        return MaTirfPipeline

    def display_window_manager_class(self):
        return DisplayWindowManager

    def create_figures_section(self):
        return FiguresSection(parent=self)

    def create_synthetic_section(self):
        return SyntheticTruthSection(self.pipeline)

    def update_figures(self, f, config):
        self.figures_section.update_plot(f, config)

    def update_synthetic(self):
        self.synthetic_section.update_plot()

    def on_close_synthetic_cleanup(self):
        if self.synthetic_section and self.synthetic_section.viewer_window:
            self.synthetic_section.viewer_window.close()
