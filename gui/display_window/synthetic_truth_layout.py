from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QGroupBox

from gui.more_widgets.depth_map_viewer import DepthMapViewer
from gui.more_widgets.histogram_3d_object import HistogramWidget
from gui.more_widgets.image_3d_viewer import Image3DViewer
from gui.more_widgets.profiles_viewer import ProfilesViewer


class SyntheticTruthSection(QGroupBox):

    def __init__(self):
        super().__init__('Synthetic Ground Truth Analysis')
        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        # layout is initially empty and widget will be add when update_plot method is called
        self.setLayout(self.layout)

    def update_plot(self, f, f_true, config):
        z0 = config['oper-params']['z0']
        zN = config['oper-params']['zN']

        f = (f - f.min()) / (f.max() - f.min())
        f_true = (f_true - f_true.min()) / (f_true.max() - f_true.min())
        diff = f_true - f
        # clear the layout:
        self.clear_layout(self.layout)
        # create the objects for the layout:
        viewer_3d = Image3DViewer(diff)
        hist = HistogramWidget(diff, bins=64)
        # build the widgets together to make the layout:
        self.layout.addWidget(viewer_3d, 2)  # 2/3 of the width
        self.layout.addWidget(hist, 1)  # 1/3 of the width


    @staticmethod
    def clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
