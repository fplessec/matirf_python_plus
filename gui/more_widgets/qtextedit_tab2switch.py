from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTextEdit, QLabel


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


if __name__=="__main__":  # test
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QLineEdit, QGroupBox, QStyleFactory

    import settings

    class TabTestWidget(QGroupBox):
        def __init__(self, title='title'):
            super().__init__(title=title)
            self.setup_ui()

        def setup_ui(self):
            layout = QVBoxLayout()
            edit1 = QLineEdit()
            edit1.setPlaceholderText("Field 1")
            label1 = QLabel("Press TAB here → should go to next widget")
            text = QTextEditTab2Switch(parent=self)
            text.setPlaceholderText("Field 2")
            label2 = QLabel("Press Shift+TAB → should go to previous widget")
            edit2 = QLineEdit()
            edit2.setPlaceholderText("Field 3")
            layout.addWidget(edit1)
            layout.addWidget(label1)
            layout.addWidget(text)
            layout.addWidget(label2)
            layout.addWidget(edit2)
            self.setLayout(layout)

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    window = TabTestWidget(title="test of object: QTextEditTab2Switch")
    window.resize(300, 200)
    window.show()

    sys.exit(app.exec_())