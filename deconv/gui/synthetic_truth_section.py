from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QHBoxLayout,
)

from gui.more_widgets import ImageAndHisto2DViewer


## popup window showing the difference image f_true - alpha*f (2D):
class DeconvDifferenceViewer(QWidget):

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


## synthetic ground truth analysis section for the deconv display window:
class DeconvSyntheticTruthSection(QGroupBox):

    def __init__(self, pipeline):
        super().__init__("Synthetic Ground Truth Analysis")
        self.viewer_window = None
        self.pipeline = pipeline
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.viewer_btn = QPushButton("Visualize difference")
        self.viewer_btn.clicked.connect(self._open_viewer)
        layout.addWidget(self.table)
        layout.addWidget(self.viewer_btn)
        self.setLayout(layout)

    def update_plot(self):
        result = self.pipeline.result
        if result is None or not result.has_synthetic_truth() or result.metrics is None:
            return
        self._populate_table(result.metrics)

    def _populate_table(self, metrics: dict):
        self.table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def _open_viewer(self):
        result = self.pipeline.result
        if result is None or result.diff is None:
            return
        if self.viewer_window is not None:
            self.viewer_window.close()
        self.viewer_window = DeconvDifferenceViewer(result.diff, parent=self)
        self.viewer_window.show()
