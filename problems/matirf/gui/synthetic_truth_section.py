from gui.specializable.display_window import BaseSyntheticTruthSection
from gui.widgets import ImageAndHisto3DViewer


class SyntheticTruthSection(BaseSyntheticTruthSection):
    """MA-TIRF synthetic-truth analysis: the difference popup uses the 3D viewer."""

    DIFFERENCE_VIEWER_CLASS = ImageAndHisto3DViewer
