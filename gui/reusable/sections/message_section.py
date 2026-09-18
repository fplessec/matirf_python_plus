from PyQt5.QtWidgets import QGroupBox, QVBoxLayout

from gui.widgets import QTextEditTab2Switch


class MessageSection(QGroupBox):

    def __init__(self, title):
        super().__init__(title)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        self.text_edit = QTextEditTab2Switch(parent=self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def _print(self, message):
        self.text_edit.append(message)
        self.text_edit.ensureCursorVisible()

    def get_text(self):
        return self.text_edit.toPlainText()


if __name__=="__main__":  # test
    import sys
    from PyQt5.QtWidgets import QApplication, QStyleFactory

    import settings as settings

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    window = MessageSection("test of object: MessageSection")
    for i in range(1, 6):
        window._print(f"log message number {i}")
    window.resize(420, 300)
    window.show()

    sys.exit(app.exec_())