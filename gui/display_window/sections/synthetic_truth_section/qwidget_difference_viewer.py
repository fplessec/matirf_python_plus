from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox

from gui.more_widgets import Image3DViewer, HistogramWidget


class DifferenceViewer(QWidget):
    def __init__(self, f, f_true, parent=None):
        super().__init__()
        self.parent = parent
        self.f = f.clone()
        self.f_true = f_true.clone()
        self.setWindowTitle("Difference Viewer")
        self.resize(700, 900)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        diff = self.f_true - self.f
        # a qgroup with Image3DViewer and HistogramWidget:
        group_diff = QGroupBox("Difference (f_true - f)")
        layout_diff = QVBoxLayout()
        viewer = Image3DViewer(diff)
        hist = HistogramWidget(diff, bins=64)
        # assemble the widgets in the layout and to the main_layout:
        layout_diff.addWidget(viewer)
        layout_diff.addWidget(hist)
        group_diff.setLayout(layout_diff)
        main_layout.addWidget(group_diff)
        self.setLayout(main_layout)

    def closeEvent(self, event):
        if self.parent:
            self.parent.viewer_window = None
        super().closeEvent(event)