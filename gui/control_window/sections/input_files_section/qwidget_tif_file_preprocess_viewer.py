from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout

from gui.more_widgets import HistogramWidget, Image3DViewer
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

        group_f = QGroupBox("tif file raw")
        layout_f = QVBoxLayout()

        viewer_f = Image3DViewer(self.f)
        hist_f = HistogramWidget(self.f)

        layout_f.addWidget(viewer_f)
        layout_f.addWidget(hist_f)
        group_f.setLayout(layout_f)

        group_g = QGroupBox("tif file preprocessed")
        layout_g = QVBoxLayout()

        viewer_g = Image3DViewer(self.g)
        hist_g = HistogramWidget(self.g)

        layout_g.addWidget(viewer_g)
        layout_g.addWidget(hist_g)
        group_g.setLayout(layout_g)

        # order :
        main_layout.addWidget(group_f)
        main_layout.addWidget(group_g)

        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.parent.tif_file_preprocess_editor = None
        super().closeEvent(event)