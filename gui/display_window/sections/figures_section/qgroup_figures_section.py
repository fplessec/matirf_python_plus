from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QStackedLayout, QWidget, QVBoxLayout

from gui.more_widgets import QSwitchButton
from gui.more_widgets.depth_map_viewer import DepthMapViewer
from gui.more_widgets.image_3d_viewer import Image3DViewer
from gui.more_widgets.profiles_viewer import ProfilesViewer
from .depth_map import DepthMap
from .profiles import Profiles
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
        config = self.parent.get_config()
        view1 = QWidget()
        layout = QHBoxLayout(view1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # creation of the widgets one after another:
        self.depth_map_widget = DepthMapViewer(z0=config['oper-params']['z0'], zN=config['oper-params']['zN'])
        self.profiles_widget = ProfilesViewer(z0=config['oper-params']['z0'], zN=config['oper-params']['zN'])
        # build the widgets together to make the layout:
        layout.addWidget(self.depth_map_widget, 2)  # 2/3 of the width
        layout.addWidget(self.profiles_widget, 1)  # 1/3 of the width
        return view1

    def create_view2_widget(self):
        view2 = QWidget()
        # creation of the widgets one after another:
        #viewer3d = Image3DViewer()
        text_layout = QVBoxLayout(view2)
        text_layout.setAlignment(Qt.AlignCenter)
        self.text_label = QLabel("texte")
        self.text_label.setAlignment(Qt.AlignCenter)
        # build the widgets together to make the layout:
        text_layout.addWidget(self.text_label)
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

    def update_plot(self):
        self.depth_map_widget.update_plot()
        self.profiles_widget.update_plot()