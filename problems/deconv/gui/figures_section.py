from gui.specializable.display_window.sections import BaseFiguresSection, View, image_panel
from gui.widgets import ImageAndHisto2DViewer


class DeconvFiguresSection(BaseFiguresSection):
    """Deconv figures: a single 2D image + histogram view."""

    VIEWS = [
        View("Reconstruction", [image_panel(ImageAndHisto2DViewer, title="Reconstruction")]),
    ]
