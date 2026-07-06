from PyQt5.QtWidgets import QWidget, QHBoxLayout

from common.gui.widgets import ImageAndHisto3DViewer
from common import DataMode
from matirf.core.pipeline_operations import MaTirfOperations
from matirf import MATIRF_CONFIG_PATH
from common.in_out import load_or_create_toml


class TifFilePreprocessViewer(QWidget):
    """
    A qwidget that allows the user to visualize two versions of its chosen input file (tif file),
    depending on the mode (REAL or SYNTHETIC):
        - REAL :      [input file = raw MA-TIRF measurement g_raw]   vs   [its preprocessed version + noise]
        - SYNTHETIC : [input file = ground truth f_true]    vs   [g_synth = H_synth @ f_true preprocessed + noise]
    """
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        config = load_or_create_toml(MATIRF_CONFIG_PATH)
        self.mode = DataMode.from_config(config['input-paths']['mode'])
        # depending on the mode we visualize:
        # left  = raw MA-TIRF measurement OR ground truth
        # right = preprocessed+noise OR synthetic MA-TIRF measurement
        self.left, self.right = MaTirfOperations.compute_preprocessing_preview(config, self.mode)
        self.setWindowTitle("Preview - Real measurement preprocessing" if self.mode == DataMode.REAL
                            else "Preview - Synthetic data simulation")
        self.resize(700, 900)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        if self.mode == DataMode.REAL:
            left_title = "g_raw = input file (MA-TIRF image stack, raw measurement)"
            right_title = "g = preprocessed g_raw + noise"
        else:  # SYNTHETIC
            left_title = "f_true = input file (3D object, synthetic truth)"
            right_title = "g_synth = H_synth @ f_true (preprocessed + noise)"
        # two ImageAndHisto3DViewer qgroups to visualize left and right:
        group_left = ImageAndHisto3DViewer(self.left, title=left_title)
        group_right = ImageAndHisto3DViewer(self.right, title=right_title)
        # assemble:
        main_layout.addWidget(group_left)
        main_layout.addWidget(group_right)
        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.parent.tif_file_preprocess_editor = None
        super().closeEvent(event)