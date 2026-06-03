from PyQt5.QtWidgets import QGroupBox, QVBoxLayout

from gui.more_widgets import ImageAndHisto2DViewer


## figures section for the deconv display window (ImageAndHisto2DViewer):
class DeconvFiguresSection(QGroupBox):

    def __init__(self, parent=None):
        super().__init__("Figures")
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def update_plot(self, f, config):
        self._clear()
        viewer = ImageAndHisto2DViewer(f, title='Reconstruction')
        self.layout.addWidget(viewer)

    def _clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
