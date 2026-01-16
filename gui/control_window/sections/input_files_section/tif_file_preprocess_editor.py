from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout

from gui.more_widgets.histogram_3d_object import HistogramWidget
from gui.more_widgets.image_3d_viewer import Image3DViewer
from in_out import load_tif, load_or_create_toml, CONFIG_PATH, load_json
from preprocess_measurement import preprocess_measurement_stack


class TifFilePreprocessEditor(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        config = load_or_create_toml(CONFIG_PATH)
        tif_path = config['input-paths']['tif']
        json_path = config['input-paths']['json']
        measurement_params = load_json(json_path)
        add_noise_params = config['add-noise']
        self.f = load_tif(tif_path)
        #self.g = self.f.clone()
        self.g, measurement_params = preprocess_measurement_stack(self.f, measurement_params, add_noise_params,
                                                                  normalization=4)

        self.setWindowTitle("Info .....")

        self.resize(700, 900)

        self.setup_ui()


    def setup_ui(self):
        main_layout = QHBoxLayout()

        # -------- Group f --------
        group_f = QGroupBox("f")
        layout_f = QVBoxLayout()

        viewer_f = Image3DViewer(self.f)
        hist_f = HistogramWidget(self.f)

        layout_f.addWidget(viewer_f)
        layout_f.addWidget(hist_f)
        group_f.setLayout(layout_f)

        # -------- Group g --------
        group_g = QGroupBox("g")
        layout_g = QVBoxLayout()

        viewer_g = Image3DViewer(self.g)
        hist_g = HistogramWidget(self.g)

        layout_g.addWidget(viewer_g)
        layout_g.addWidget(hist_g)
        group_g.setLayout(layout_g)

        # ordre :
        main_layout.addWidget(group_f)
        main_layout.addWidget(group_g)

        self.setLayout(main_layout)

    # def setup_ui(self):
    #     layout = QVBoxLayout()
    #     viewer = Image3DViewer(self.f)
    #     hist = HistogramWidget(self.f)
    #     layout.addWidget(viewer)
    #     layout.addWidget(hist)
    #     self.setLayout(layout)



    def closeEvent(self, event):
        self.parent.tif_file_preprocess_editor = None
        super().closeEvent(event)