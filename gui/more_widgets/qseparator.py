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


if __name__=="__main__":  # test
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QStyleFactory

    import settings

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    window = QGroupBox(title="test of object: QSeparator")
    layout = QVBoxLayout()

    layout.addWidget(QLabel("Above horizontal separator"))
    layout.addWidget(QSeparator('h'))
    layout.addWidget(QLabel("Below horizontal separator"))

    layoutH = QHBoxLayout()
    layoutH.addStretch()
    layoutH.addWidget(QSeparator('v'))
    layoutH.addStretch()
    layout.addLayout(layoutH)

    window.setLayout(layout)
    window.resize(300, 200)
    window.show()

    sys.exit(app.exec_())