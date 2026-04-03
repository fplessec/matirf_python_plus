from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout

from gui.more_widgets import ImageAndHisto3DViewer
from core.preprocess_measurement import preprocess_measurement_stack
from in_out import load_tif, load_or_create_toml, CONFIG_PATH, load_json
import settings


class TifFilePreprocessViewer(QWidget):
    """
    A qwidget that allows the user to see the difference between its chosen raw data (tif file) and the preprocessed
    version of this raw data that will be used for the reconstruction.
    """
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        config = load_or_create_toml(CONFIG_PATH)
        tif_path = config['input-paths']['tif']
        json_path = config['input-paths']['json']
        measurement_params = load_json(json_path)
        add_noise_params = config['add-noise']
        self.f = load_tif(tif_path)
        self.g, measurement_params = preprocess_measurement_stack(self.f, measurement_params, add_noise_params,
                                                                  normalization=settings.normalization)
        self.setWindowTitle("See preprocessed file")
        self.resize(700, 900)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        # a qgroup with ImageAndHisto3DViewer:
        group_f = ImageAndHisto3DViewer(self.f, title="tif file raw")
        # a qgroup with ImageAndHisto3DViewer:
        group_g = ImageAndHisto3DViewer(self.g, title="tif file preprocessed + add noise")
        # assemble :
        main_layout.addWidget(group_f)
        main_layout.addWidget(group_g)
        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.parent.tif_file_preprocess_editor = None
        super().closeEvent(event)