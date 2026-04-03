from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPainter, QPen, QPalette
from PyQt5.QtCore import Qt


class QCrossButton(QPushButton):
    """
    A minimal button displaying a cross (X), typically used for closing or removing elements:
        > renders a custom cross icon using QPainter
        > adapts its colors to the current Qt palette
        > provides hover feedback and a compact fixed size
        > intended for lightweight UI actions such as delete/close
    """
    def __init__(self, size=12):
        super().__init__()
        self.qt_color = self.palette().color(QPalette.PlaceholderText) # the color depends on the QPalette
        self.size = size
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        background_color = self.palette().color(QPalette.WindowText).name() # the color depends on the QPalette
        self.setStyleSheet(f"""QPushButton {{font-weight: bold;
                                           border: 1px solid #666;
                                           border-radius: 3px;
                                           padding: 0;}}
                              QPushButton:hover {{background-color: {background_color};}}""")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(self.qt_color, 2)
        painter.setPen(pen)
        m = 4  # margin inside the square
        painter.drawLine(m, m, self.size - m, self.size - m)
        painter.drawLine(self.size - m, m, m, self.size - m)


if __name__=="__main__":  # test
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QGroupBox, QLabel, QStyleFactory

    import settings

    class CrossTestWidget(QGroupBox):
        def __init__(self, title='title'):
            super().__init__(title=title)
            self.setup_ui()

        def setup_ui(self):
            layout = QVBoxLayout()
            self.label = QLabel("Click the cross")
            self.cross = QCrossButton(size=20)
            self.cross.clicked.connect(self.on_click)
            layout.addWidget(self.cross)
            layout.addWidget(self.label)
            self.setLayout(layout)

        def on_click(self):
            self.label.setText("Cross clicked")

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    window = CrossTestWidget(title="test of object: QCrossButton")
    window.resize(200, 100)
    window.show()

    sys.exit(app.exec_())