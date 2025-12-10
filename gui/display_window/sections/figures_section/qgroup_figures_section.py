from PyQt5.QtWidgets import QGroupBox, QHBoxLayout

from .depth_map import DepthMap
from .profiles import Profiles


class FiguresSection(QGroupBox):

    def __init__(self):
        super().__init__("Figures")
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        self.depth_map_widget = DepthMap()
        self.profiles_widget = Profiles()

        layout.addWidget(self.depth_map_widget, 2)  # 2 tiers de la largeur
        layout.addWidget(self.profiles_widget, 1)  # 1 tier de la largeur
        self.setLayout(layout)
