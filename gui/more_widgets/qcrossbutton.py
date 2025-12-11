from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt


class QCrossButton(QPushButton):
    def __init__(self, size=12):
        super().__init__()
        self.size = size
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""QPushButton {font-weight: bold;
                                           border: 1px solid #666;
                                           border-radius: 3px;
                                           padding: 0;}
                              QPushButton:hover {background-color: #eee;}""")
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.white, 2)
        painter.setPen(pen)
        m = 4  # margin inside the square
        painter.drawLine(m, m, self.size - m, self.size - m)
        painter.drawLine(self.size - m, m, m, self.size - m)