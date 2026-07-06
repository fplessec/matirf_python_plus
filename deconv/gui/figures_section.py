from PyQt5.QtWidgets import QVBoxLayout, QWidget

from common.gui.specializable.display_window.sections import BaseFiguresSection
from common.gui.widgets import ImageAndHisto2DViewer


class DeconvFiguresSection(BaseFiguresSection):
    """
    Figures section for the deconv display window.
    Single view: ImageAndHisto2DViewer for 2D deconvolution results.
    """

    def __init__(self, parent=None):
        ## viewer reference (populated on first create_views call):
        self.viewer_2d = None
        super().__init__(parent)

    def create_views(self):
        ## single view: 2D image viewer with histogram:
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        self.viewer_2d = ImageAndHisto2DViewer(self._init_f, title='Reconstruction')
        view_layout.addWidget(self.viewer_2d)
        return [view]

    def update_views(self, f, config):
        ## store data for initial view creation:
        self._init_f = f
        ## if viewer is already created, update it in-place:
        if self.viewer_2d is not None:
            self.viewer_2d.set_image(f)
