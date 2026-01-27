from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPainter, QPen, QPalette
from PyQt5.QtCore import Qt


class QCrossButton(QPushButton):
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