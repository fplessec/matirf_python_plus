from common.gui.specializable.display_window import BaseSyntheticTruthSection
from common.gui.widgets import ImageAndHisto2DViewer


class DeconvSyntheticTruthSection(BaseSyntheticTruthSection):
    """Deconv synthetic-truth analysis: the difference popup uses the 2D viewer."""

    DIFFERENCE_VIEWER_CLASS = ImageAndHisto2DViewer
