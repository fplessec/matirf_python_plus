from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from .qwidget_difference_viewer import DifferenceViewer


class SyntheticTruthSection(QGroupBox):

    def __init__(self, pipeline):
        super().__init__("Synthetic Ground Truth Analysis")
        self.viewer_window = None
        self.pipeline = pipeline
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout()

        # TABLE
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # BUTTON
        self.viewer_btn = QPushButton("Visualize difference")
        self.viewer_btn.clicked.connect(self.open_viewer)

        self.layout.addWidget(self.table)
        self.layout.addWidget(self.viewer_btn)

        self.setLayout(self.layout)


    def update_plot(self):
        # Le pipeline garantit qu'en mode synth, result.metrics est calculé
        # dans _on_algo_finished. Plus de fallback de calcul ici (qui était
        # à la fois buggé et hors-sujet pour un widget).
        result = self.pipeline.result
        if result is None or not result.has_synthetic_truth() or result.metrics is None:
            return
        self.populate_table(result.metrics)

    def populate_table(self, metrics: dict):
        self.table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def open_viewer(self):
        # La diff est précalculée par le pipeline et stockée dans result.diff.
        result = self.pipeline.result
        if result is None or result.diff is None:
            return
        # to avoid having more than one window:
        if self.viewer_window is not None:
            self.viewer_window.close()
        self.viewer_window = DifferenceViewer(result.diff, parent=self)
        self.viewer_window.show()