from common.gui.specializable.display_window import BaseSyntheticTruthSection
from .qwidget_difference_viewer import DifferenceViewer


class SyntheticTruthSection(BaseSyntheticTruthSection):

    def _create_difference_viewer(self, diff):
        return DifferenceViewer(diff, parent=self)
