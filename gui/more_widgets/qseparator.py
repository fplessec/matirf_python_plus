from PyQt5.QtWidgets import QFrame


class QSeparator(QFrame):
    """A widget to draw a line for style. Vertically or horizontally."""
    def __init__(self, shape):
        super().__init__()
        if shape in ['v', 'V']:
            self.setFrameShape(QFrame.VLine)
        elif shape in ['h', 'H']:
            self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)