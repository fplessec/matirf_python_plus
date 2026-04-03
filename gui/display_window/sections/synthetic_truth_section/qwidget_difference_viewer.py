from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox

from gui.more_widgets import ImageAndHisto3DViewer


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

        from core.reconstruction_metrics import optimal_scale
        from core.operations import estimate_delta_anisotropy_from_params
        from in_out import load_json, load_or_create_toml, CACHE_DIR
        from os.path import join
        config = load_or_create_toml(join(CACHE_DIR, 'config.toml'))
        measurement_params = load_json(config['input-paths']['json'])
        operator_params = config['oper-params']
        delta = estimate_delta_anisotropy_from_params(measurement_params, operator_params)
        alpha = optimal_scale(self.f.detach().cpu().numpy(), self.f_true.detach().cpu().numpy(), delta=delta)

        print('alpha U.I:', alpha)

        diff = self.f_true - alpha * self.f

        qgroup_viewer = ImageAndHisto3DViewer(image=diff, title="Difference (f_true - alpha * f)")
        main_layout.addWidget(qgroup_viewer)
        self.setLayout(main_layout)

    def closeEvent(self, event):
        if self.parent:
            self.parent.viewer_window = None
        super().closeEvent(event)