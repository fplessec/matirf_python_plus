from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QGridLayout
from PyQt5.QtCore import pyqtSignal, Qt


class QSwitchButton(QWidget):
    """
    A custom toggle switch widget that mimics an on/off slider:
        > displays a movable circular button inside a rounded container
        > toggles state on click, switching position (left/right) and color
        > emits a boolean signal (toggled) when the state changes
        > provides methods to programmatically switch state (left/right)

    I rewrote this code https://pypi.org/project/pyqt-switch/ from  Jung Gyu Yoon.
    """
    toggled = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.circle_diameter = 20
        self._positioned_to_left = True
        self.setup_ui()

    def setup_ui(self):
        # a circle:
        self.circle = QPushButton()
        self.circle.setCheckable(True)
        self.circle.toggled.connect(self._toggled)
        self.set_color(255)
        # switch button layout:
        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.circle)
        self.layout.setContentsMargins(0, 0, 0, 0)
        # inner widget:
        innerWidgetForStyle = QWidget()
        innerWidgetForStyle.setLayout(self.layout)
        # main layout:
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
            if no_signal:
                self.blockSignals(True)
            self.circle.toggle()
            if no_signal:
                self.blockSignals(False)

    def switch_to_right(self, no_signal=False):
        if self._positioned_to_left:
            if no_signal:
                self.blockSignals(True)
            self.circle.toggle()
            if no_signal:
                self.blockSignals(False)


if __name__=="__main__":  # test
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QLabel, QGroupBox, QPushButton, QStyleFactory

    import settings

    class SwitchTestWidget(QGroupBox):
        def __init__(self, title='title'):
            super().__init__(title=title)
            self.setup_ui()

        def setup_ui(self):
            layout = QVBoxLayout()
            self.label = QLabel("State: OFF")
            self.switch = QSwitchButton()
            self.switch.toggled.connect(self.on_toggle)
            btn_left = QPushButton("force left")
            btn_left.clicked.connect(lambda: self.switch.switch_to_left())
            btn_right = QPushButton("force right")
            btn_right.clicked.connect(lambda: self.switch.switch_to_right())
            layout.addWidget(self.switch)
            layout.addWidget(self.label)
            layout.addWidget(btn_left)
            layout.addWidget(btn_right)
            self.setLayout(layout)

        def on_toggle(self, state):
            self.label.setText(f"State: {'ON' if state else 'OFF'}")


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    window = SwitchTestWidget(title="test of object: QSwitchButton")
    window.resize(200, 200)
    window.show()

    sys.exit(app.exec_())