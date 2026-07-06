from common.gui.specializable.display_window import BaseDisplayWindow
from .display_window_manager import DeconvDisplayWindowManager
from .figures_section import DeconvFiguresSection
from .synthetic_truth_section import DeconvSyntheticTruthSection
from deconv.core import DeconvPipeline
from deconv import DECONV_RESULTS_DIR


class DeconvDisplayWindow(BaseDisplayWindow):

    def window_title(self):
        return "Deconv Display Window"

    def results_dir(self):
        return DECONV_RESULTS_DIR

    def pipeline_class(self):
        return DeconvPipeline

    def display_window_manager_class(self):
        return DeconvDisplayWindowManager

    def create_figures_section(self):
        return DeconvFiguresSection(parent=self)

    def create_synthetic_section(self):
        return DeconvSyntheticTruthSection(self.pipeline)

    def update_figures(self, f, config):
        self.figures_section.update_plot(f, config)

    def update_synthetic(self):
        self.synthetic_section.update_plot()

    def on_close_synthetic_cleanup(self):
        if self.synthetic_section and self.synthetic_section.viewer_window:
            self.synthetic_section.viewer_window.close()
