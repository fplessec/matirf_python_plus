from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTextEdit


class QTextEditTab2Switch(QTextEdit):
    """A QTextEdit where the 'Tab' touch can't write anything, but will switch to the next QWidget from its layout."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.parent.focusNextChild()  # Tab to go to next widget
        elif event.key() == Qt.Key_Backtab:
            self.parent.focusPreviousChild()  # Shift+Tab to go back
        else:
            super().keyPressEvent(event)