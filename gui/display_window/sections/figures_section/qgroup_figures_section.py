from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QStackedLayout, QWidget, QVBoxLayout

from gui.more_widgets import QSwitchButton
from gui.more_widgets.depth_map_viewer import DepthMapViewer
from gui.more_widgets.histogram_3d_object import HistogramWidget
from gui.more_widgets.image_3d_viewer import Image3DViewer
from gui.more_widgets.profiles_viewer import ProfilesViewer
import settings


class FiguresSection(QGroupBox):

    def __init__(self, parent):
        super().__init__("Figures")
        self.parent = parent
        self.is_view1 = True  # default state: view 1
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        # creation of the widgets one after another:
        header = self.create_header_widget()
        self.stacked_layout = QStackedLayout()
        view1 = self.create_view1_widget()
        view2 = self.create_view2_widget()
        self.stacked_layout.addWidget(view1)
        self.stacked_layout.addWidget(view2)
        # build the widgets together to make the layout:
        layout.addWidget(header)
        layout.addLayout(self.stacked_layout)
        self.setLayout(layout)

    def create_header_widget(self):
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(6)
        # creation of the widgets one after another:
        self.label1 = QLabel("view 1")
        self.label2 = QLabel("view 2")
        self.update_labels()
        self.switch = QSwitchButton()
        self.switch.toggled.connect(self.on_switch_toggled)
        # build the widgets together to make the layout:
        layout.addWidget(self.label1)
        layout.addWidget(self.switch)
        layout.addWidget(self.label2)
        layout.addStretch()
        return header

    def create_view1_widget(self):
        view1 = QWidget()
        self.view1_layout = QHBoxLayout(view1)
        self.view1_layout.setContentsMargins(0, 0, 0, 0)
        self.view1_layout.setSpacing(0)
        # view1 is initially empty and widget will be add when update_plot method is called
        return view1

    def create_view2_widget(self):
        view2 = QWidget()
        self.view2_layout = QVBoxLayout(view2)
        self.view2_layout.setContentsMargins(0, 0, 0, 0)
        self.view2_layout.setSpacing(0)
        # view2 is initially empty and widget will be add when update_plot method is called
        return view2

    def on_switch_toggled(self):
        self.is_view1 = not self.is_view1
        self.stacked_layout.setCurrentIndex(0 if self.is_view1 else 1)
        self.update_labels()

    def update_labels(self):
        if self.is_view1:
            self.label1.setStyleSheet(f"color: white; font-size: {settings.FontSize.SMALL}pt;")
            self.label2.setStyleSheet(f"color: gray; font-size: {settings.FontSize.SMALL}pt;")
        else:
            self.label2.setStyleSheet(f"color: white; font-size: {settings.FontSize.SMALL}pt;")
            self.label1.setStyleSheet(f"color: gray; font-size: {settings.FontSize.SMALL}pt;")

    def set_image(self, image):
        self.depth_map_widget.set_image(image)
        self.profiles_widget.set_image(image)

    def update_plot(self, f, config):
        z0 = config['oper-params']['z0']
        zN = config['oper-params']['zN']
        # clear the view1 layout:
        self.clear_layout(self.view1_layout)
        # create the objects for view1:
        depth_map_widget = DepthMapViewer(f, z0, zN)
        profiles_widget = ProfilesViewer(f, z0, zN)
        # build the widgets together to make the view1 layout:
        self.view1_layout.addWidget(depth_map_widget, 2)  # 2/3 of the width
        self.view1_layout.addWidget(profiles_widget, 1)  # 1/3 of the width
        # clear the view2 layout:
        self.clear_layout(self.view2_layout)
        # create the objects for view1:
        viewer_3d = Image3DViewer(f)
        hist = HistogramWidget(f, bins=64)
        # build the widgets together to make the view1 layout:
        self.view2_layout.addWidget(viewer_3d, 2)  # 2/3 of the height
        self.view2_layout.addWidget(hist, 1)  # 1/3 of the height

    @staticmethod
    def clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()