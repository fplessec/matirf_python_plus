"""
MA-TIRF figures section, declared as two views:
    > view 1: DepthMapViewer (2/3) + ProfilesViewer (1/3), side by side, with PNG export
    > view 2: ImageAndHisto3DViewer (full)
The depth-map / profiles viewers need z0 / zN (pulled from the config), hence the small
build/refresh helpers below; the 3D view is the generic image_panel case.
"""

from gui.specializable.display_window.sections import BaseFiguresSection, View, Panel, image_panel
from gui.widgets import DepthMapViewer, ImageAndHisto3DViewer, ProfilesViewer
from gui.widgets.figure_export import export_depth_and_profiles
from problems.matirf import MATIRF_RESULTS_DIR


def _z_bounds(config):
    return config['oper-params']['z0'], config['oper-params']['zN']


def _build_depth_map(f, config):
    z0, zN = _z_bounds(config)
    return DepthMapViewer(f, z0, zN)


def _build_profiles(f, config):
    z0, zN = _z_bounds(config)
    return ProfilesViewer(f, z0, zN)


def _refresh_z_view(widget, f, config):
    """Refresh a DepthMap/Profiles viewer: re-apply z bounds then the image."""
    widget.z0, widget.zN = _z_bounds(config)
    widget.set_image(f)


def _export_view1(view_widgets, filepath):
    """view_widgets = [depth_map_viewer, profiles_viewer]."""
    export_depth_and_profiles(view_widgets[0], view_widgets[1], filepath)


class FiguresSection(BaseFiguresSection):

    EXPORT_DEFAULT_DIR = MATIRF_RESULTS_DIR

    VIEWS = [
        View("view 1",
             [Panel(_build_depth_map, _refresh_z_view, stretch=2),
              Panel(_build_profiles, _refresh_z_view, stretch=1)],
             export=_export_view1),
        View("view 2",
             [image_panel(ImageAndHisto3DViewer, title="")]),
    ]
