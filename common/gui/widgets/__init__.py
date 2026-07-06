"""
This module gathers all custom PyQt widgets used across the graphical user interface of the application.

It includes both general-purpose UI components (buttons, labels, separators, text inputs) and specialized
visualization widgets designed for exploring and analyzing 2D and 3D data.

I can describe the objects in this module along those three main categories:
    > generic UI widgets:
        - QLatexLabel: renders LaTeX expressions as images inside Qt widgets
        - QSeparator: simple horizontal/vertical separator for layouts
        - QTextEditTab2Switch: text edit with custom Tab navigation behavior
        - QSwitchButton: toggle switch widget (on/off)
        - QCrossButton: small cross button for close/remove actions
    > 2D visualization widgets:
        - Image2DViewer: interactive viewer for a 2D image with pixel inspection
        - Histogram2DWidget: histogram visualization of 2D data with Full and XY-patch modes
        - ImageAndHisto2DViewer: high-level widget combining 2D image viewer + histogram with controls
    > 3D visualization widgets:
        - Image3DViewer: interactive viewer to navigate through Z-slices of a 3D image
        - Histogram3DWidget: histogram visualization of 3D data with multiple modes
        - ImageAndHisto3DViewer: high-level widget combining image viewer + histogram with controls
        - DepthMapViewer: projection-based visualization of depth information
        - ProfilesViewer: orthogonal projections (yz, zx) to analyze structural profiles
"""

# user interface specific widgets:
from .qlatexlabel import QLatexLabel
from .qseparator import QSeparator
from .qtextedit_tab2switch import QTextEditTab2Switch
from .qswitchbutton import QSwitchButton
from .qcrossbutton import QCrossButton
# 2d visualisation specific widgets:
from .image_2d_viewer import Image2DViewer
from .histogram_2d_widget import Histogram2DWidget
from .image_and_histo_2d_viewer import ImageAndHisto2DViewer
# 3d visualisation specific widgets:
from .depth_map_viewer import DepthMapViewer
from .profiles_viewer import ProfilesViewer
from .histogram_3d_widget import Histogram3DWidget
from .image_3d_viewer import Image3DViewer
from .image_and_histo_3d_viewer import ImageAndHisto3DViewer