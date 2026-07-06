from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from common.gui.widgets import ImageAndHisto3DViewer


class DifferenceViewer(QWidget):
    """
    Displays the difference f_true - alpha*f.

    NOTE: the diff is precomputed by the pipeline (see _on_algo_finished
    in synthetic mode) and stored in pipeline.result.diff. This viewer
    only displays it, no computation is done here.
    """
    def __init__(self, diff, parent=None):
        super().__init__()
        self.parent = parent
        self.diff = diff.clone()
        self.setWindowTitle("Difference Viewer")
        self.resize(700, 900)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        qgroup_viewer = ImageAndHisto3DViewer(
            image=self.diff,
            title="Difference (f_true - alpha * f)"
        )
        main_layout.addWidget(qgroup_viewer)
        self.setLayout(main_layout)

    def closeEvent(self, event):
        if self.parent:
            self.parent.viewer_window = None
        super().closeEvent(event)