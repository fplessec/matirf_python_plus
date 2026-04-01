from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QGridLayout
from PyQt5.QtCore import pyqtSignal, Qt


class QSwitchButton(QWidget):
    """
    I rewrote this code https://pypi.org/project/pyqt-switch/ from  Jung Gyu Yoon.
    """
    toggled = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.circle_diameter = 20
        self._positioned_to_left = True
        self.setup_ui()

    def setup_ui(self):
        self.circle = QPushButton()
        self.circle.setCheckable(True)
        self.circle.toggled.connect(self._toggled)
        self.set_color(255)

        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.circle)
        self.layout.setContentsMargins(0, 0, 0, 0)

        innerWidgetForStyle = QWidget()
        innerWidgetForStyle.setLayout(self.layout)

        main_layout = QGridLayout()
        main_layout.addWidget(innerWidgetForStyle)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(main_layout)
        self.set_style()

    def set_style(self):
        self.circle.setFixedSize(self.circle_diameter, self.circle_diameter)
        self.setStyleSheet(
            f'QWidget {{ border: {self.circle_diameter // 20}px solid #AAAAAA; '
            f'border-radius: {self.circle_diameter // 2}px; }}')
        self.setFixedSize(self.circle_diameter * 2, self.circle_diameter)

    def mousePressEvent(self, e):
        self.circle.toggle()
        return super().mousePressEvent(e)

    def _toggled(self, f):
        if f:
            self.circle.move(self.circle_diameter, 0)
            self.layout.setAlignment(Qt.AlignRight)
            self.set_color(200)
            self._positioned_to_left = False
        else:
            self.circle.move(0, 0)
            self.layout.setAlignment(Qt.AlignLeft)
            self.set_color(255)
            self._positioned_to_left = True
        self.toggled.emit(f)

    def set_color(self, f: int):
        self.circle.setStyleSheet(f'QPushButton {{ background-color: rgb({f}, {f}, 255); }}')

    def set_circle_diameter(self, diameter: int):
        self.circle_diameter = diameter
        self.set_style()

    def switch_to_left(self, no_signal=False):
        if not self._positioned_to_left:
            if no_signal: self.blockSignals(True)
            self.circle.toggle()
            if no_signal: self.blockSignals(False)
    def switch_to_right(self, no_signal=False):
        if self._positioned_to_left:
            if no_signal: self.blockSignals(True)
            self.circle.toggle()
            if no_signal: self.blockSignals(False)