from PyQt5.QtWidgets import QGroupBox, QVBoxLayout

from gui.more_widgets import QTextEditTab2Switch


class MessageSection(QGroupBox):

    def __init__(self, title):
        super().__init__(title)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        # Zone de texte en lecture seule
        self.text_edit = QTextEditTab2Switch(parent=self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def _print(self, message):
        self.text_edit.append(message)  # Ajoute une nouvelle ligne
        self.text_edit.ensureCursorVisible()  # Fait défiler automatiquement

    def get_text(self):
        return self.text_edit.toPlainText()