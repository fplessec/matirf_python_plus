from PyQt5.QtWidgets import QWidget, QHBoxLayout

from common.gui.widgets import ImageAndHisto2DViewer
from common.gui.specializable.display_window import BaseSyntheticTruthSection


class DeconvDifferenceViewer(QWidget):
    """Popup window showing the difference image f_true - alpha*f (2D)."""

    def __init__(self, diff, parent=None):
        super().__init__()
        self.parent = parent
        self.diff = diff.clone()
        self.setWindowTitle("Difference Viewer")
        self.resize(700, 900)
        layout = QHBoxLayout()
        viewer = ImageAndHisto2DViewer(
            image=self.diff,
            title="Difference (f_true - alpha * f)")
        layout.addWidget(viewer)
        self.setLayout(layout)

    def closeEvent(self, event):
        if self.parent:
            self.parent.viewer_window = None
        super().closeEvent(event)


class DeconvSyntheticTruthSection(BaseSyntheticTruthSection):

    def _create_difference_viewer(self, diff):
        return DeconvDifferenceViewer(diff, parent=self)
