"""
Abstract base class for the synthetic truth section of any inverse problem.

Provides:
    - A metrics table (Metric / Value) with adaptive display:
        * scalar metrics: displayed as a number
        * curve metrics: displayed as summary value + a button to open a plot
    - A "Visualize difference" button that opens a viewer window

Declarative model: a concrete section just declares the viewer class used for the
"difference" popup:
    DIFFERENCE_VIEWER_CLASS = ImageAndHisto2DViewer   # or ...3DViewer
The difference popup itself is generic (DifferenceViewer below).
"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QWidget, QHBoxLayout,
)

from core.metrics import METRIC_REGISTRY


class DifferenceViewer(QWidget):
    """
    Generic popup window showing the difference image f_true - alpha*f.

    The diff is precomputed by the pipeline (pipeline.result.diff); this only displays it,
    using the viewer class passed in (ImageAndHisto2DViewer / ImageAndHisto3DViewer).
    """

    def __init__(self, diff, viewer_class, parent=None):
        super().__init__()
        self.parent = parent
        self.diff = diff.clone()
        self.setWindowTitle("Difference Viewer")
        self.resize(700, 900)
        layout = QHBoxLayout()
        layout.addWidget(viewer_class(image=self.diff, title="Difference (f_true - alpha * f)"))
        self.setLayout(layout)

    def closeEvent(self, event):
        if self.parent:
            self.parent.viewer_window = None
        super().closeEvent(event)


class CurveViewerWindow(QWidget):
    """Small popup window displaying a metric curve via matplotlib."""

    def __init__(self, name, curve_data, parent=None):
        super().__init__()
        self.setWindowTitle(f"{name}")
        self.resize(500, 350)
        self._parent_section = parent
        self._build_plot(name, curve_data)

    def _build_plot(self, name, curve_data):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(5, 3.5), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(curve_data["x"], curve_data["y"], '-o', markersize=3)
        ax.set_xlabel(curve_data.get("xlabel", "x"))
        ax.set_ylabel(curve_data.get("ylabel", "y"))
        ax.set_title(f"{name}  (mean = {curve_data['summary']:.4g})")
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasQTAgg(fig)
        layout = QHBoxLayout()
        layout.addWidget(canvas)
        self.setLayout(layout)

    def closeEvent(self, event):
        if self._parent_section is not None:
            self._parent_section._curve_windows.discard(self)
        super().closeEvent(event)


class BaseSyntheticTruthSection(QGroupBox):
    """
    Synthetic-truth analysis section — a metrics table (scalar values + curve popups) and a
    "Visualize difference" button. Declared by a single class attribute.

    ----------
    > Parameters (override as class attributes) :
    ----------

    >> DIFFERENCE_VIEWER_CLASS : type
        The image viewer used for the difference popup (f_true - alpha*f), e.g.
        ImageAndHisto2DViewer / ImageAndHisto3DViewer. The popup itself is generic
        (DifferenceViewer above).

    ----------
    > Example :
    ----------

        class SyntheticTruthSection(BaseSyntheticTruthSection):
            DIFFERENCE_VIEWER_CLASS = ImageAndHisto3DViewer
    """

    # a concrete section only declares the viewer class for the difference popup:
    DIFFERENCE_VIEWER_CLASS = None   # ImageAndHisto2DViewer / ImageAndHisto3DViewer

    def __init__(self, pipeline):
        super().__init__("Synthetic Ground Truth Analysis")
        self.viewer_window = None
        self._curve_windows = set()
        self.pipeline = pipeline
        self._setup_ui()

    # -- difference popup (generic; override only for a fully custom window) --

    def _create_difference_viewer(self, diff):
        """Return the difference-image window (generic, built from DIFFERENCE_VIEWER_CLASS)."""
        return DifferenceViewer(diff, self.DIFFERENCE_VIEWER_CLASS, parent=self)

    # -- shared UI construction -----------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", ""])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.viewer_btn = QPushButton("Visualize difference")
        self.viewer_btn.clicked.connect(self._open_viewer)
        layout.addWidget(self.table)
        layout.addWidget(self.viewer_btn)
        self.setLayout(layout)

    # -- shared logic ---------------------------------------------------------

    def update_plot(self):
        result = self.pipeline.result
        if result is None or not result.has_synthetic_truth() or result.metrics is None:
            return
        self._populate_table(result.metrics)

    def _populate_table(self, metrics: dict):
        self.table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.table.setItem(row, 0, QTableWidgetItem(str(key)))
            metric_obj = METRIC_REGISTRY.get(key)
            is_curve = (metric_obj is not None and metric_obj.result_type == "curve"
                        and isinstance(value, dict))
            if is_curve:
                display_value = f"{value.get('summary', '—'):.4g}"
                self.table.setItem(row, 1, QTableWidgetItem(display_value))
                btn = QPushButton("\U0001F4C8")  # 📈
                btn.setFixedWidth(36)
                btn.setToolTip(f"View {key} curve")
                curve_data = value
                btn.clicked.connect(lambda checked, n=key, d=curve_data: self._open_curve(n, d))
                self.table.setCellWidget(row, 2, btn)
            else:
                if isinstance(value, float):
                    display_value = f"{value:.4g}"
                else:
                    display_value = str(value)
                self.table.setItem(row, 1, QTableWidgetItem(display_value))

    def _open_curve(self, name, curve_data):
        win = CurveViewerWindow(name, curve_data, parent=self)
        self._curve_windows.add(win)
        win.show()

    def _open_viewer(self):
        result = self.pipeline.result
        if result is None or result.diff is None:
            return
        if self.viewer_window is not None:
            self.viewer_window.close()
        self.viewer_window = self._create_difference_viewer(result.diff)
        self.viewer_window.show()
