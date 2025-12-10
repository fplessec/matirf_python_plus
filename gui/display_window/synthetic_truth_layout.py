from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QGroupBox


class SyntheticTruthSection(QGroupBox):

    def __init__(self):
        super().__init__('Synthetic Ground Truth Analysis')
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        # ---- Exemple de contenu ----
        label = QLabel("test de text")
        layout.addWidget(label)

        # Bouton pour revenir à la reconstruction
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()

        layout.addStretch()
        layout.addLayout(bottom_bar)

        self.setLayout(layout)
