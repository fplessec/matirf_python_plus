from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from common.gui.specializable.display_window.sections import BaseFiguresSection
from common.gui.widgets import DepthMapViewer, ImageAndHisto3DViewer, ProfilesViewer


class FiguresSection(BaseFiguresSection):
    """
    Figures section for the matirf display window.
    Two switchable views:
        > view 1: DepthMapViewer + ProfilesViewer (side by side)
        > view 2: ImageAndHisto3DViewer (full height)
    """

    def __init__(self, parent=None):
        ## viewer references (populated on first create_views call):
        self.depth_map_widget = None
        self.profiles_widget = None
        self.viewer_3d = None
        super().__init__(parent)

    def view_labels(self):
        return ["view 1", "view 2"]

    def create_views(self):
        ## view 1: depth map + profiles side by side:
        view1 = QWidget()
        view1_layout = QHBoxLayout(view1)
        view1_layout.setContentsMargins(0, 0, 0, 0)
        view1_layout.setSpacing(0)
        self.depth_map_widget = DepthMapViewer(self._init_f, self._z0, self._zN)
        self.profiles_widget = ProfilesViewer(self._init_f, self._z0, self._zN)
        view1_layout.addWidget(self.depth_map_widget, 2)  # 2/3 of the width
        view1_layout.addWidget(self.profiles_widget, 1)  # 1/3 of the width
        ## view 2: 3D image viewer with histogram:
        view2 = QWidget()
        view2_layout = QVBoxLayout(view2)
        view2_layout.setContentsMargins(0, 0, 0, 0)
        view2_layout.setSpacing(0)
        self.viewer_3d = ImageAndHisto3DViewer(self._init_f, title='')
        view2_layout.addWidget(self.viewer_3d)
        return [view1, view2]

    # ── view-1 PNG export (depth map + profiles, fixed size across windows) ──

    def supports_view1_export(self):
        return True

    def export_view1(self, filepath):
        from common.gui.widgets.figure_export import export_depth_and_profiles
        export_depth_and_profiles(self.depth_map_widget, self.profiles_widget, filepath)

    def export_default_dir(self):
        from matirf import MATIRF_RESULTS_DIR
        return str(MATIRF_RESULTS_DIR)

    def update_views(self, f, config):
        z0 = config['oper-params']['z0']
        zN = config['oper-params']['zN']
        ## store config values for initial view creation:
        self._z0 = z0
        self._zN = zN
        self._init_f = f
        ## if viewers are already created, update only the active view in-place:
        if self.depth_map_widget is not None:
            if self._current_view_index == 0:
                self.depth_map_widget.z0 = z0
                self.depth_map_widget.zN = zN
                self.depth_map_widget.set_image(f)
                self.profiles_widget.z0 = z0
                self.profiles_widget.zN = zN
                self.profiles_widget.set_image(f)
            else:
                self.viewer_3d.set_image(f)
