from PyQt5.QtWidgets import QWidget, QHBoxLayout

from common.gui.widgets import ImageAndHisto2DViewer
from common import DataMode
from deconv.core.pipeline_operations import DeconvOperations
from deconv import DECONV_CONFIG_PATH, DEFAULT_DECONV_CONFIG
from common.in_out import load_or_create_toml


## preview window showing input image vs preprocessed/synthetic measurement (2D):
class DeconvPreprocessViewer(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        config = load_or_create_toml(DECONV_CONFIG_PATH)
        self.mode = DataMode.from_config(config['input-paths']['mode'])
        self.left, self.right = DeconvOperations.compute_preprocessing_preview(config, self.mode)
        self.setWindowTitle(
            "Preview - Real measurement preprocessing" if self.mode == DataMode.REAL
            else "Preview - Synthetic data simulation")
        self.resize(900, 900)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout()
        if self.mode == DataMode.REAL:
            left_title = "g_raw = input image (raw measurement)"
            right_title = "g = preprocessed g_raw + noise"
        else:
            left_title = "f_true = input image (ground truth)"
            right_title = "g_synth = H * f_true (+ noise)"
        group_left = ImageAndHisto2DViewer(self.left, title=left_title)
        group_right = ImageAndHisto2DViewer(self.right, title=right_title)
        layout.addWidget(group_left)
        layout.addWidget(group_right)
        self.setLayout(layout)

    def closeEvent(self, event):
        self.parent.preprocess_viewer = None
        super().closeEvent(event)
