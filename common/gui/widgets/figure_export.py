"""
Fixed-size PNG export of a "view 1" (depth map + profiles).

The point is reproducibility across windows of different sizes: whatever the on-screen
size of the figures group box (e.g. the reconstruction display vs the synthetic
ground-truth generator), the saved PNG always has the SAME pixel size, the SAME image
ratios and the SAME spacing between the depth map and the profiles.

Layout (fixed, in pixels): the depth map occupies 2/3 of the content width, the profiles
1/3. The depth map keeps its true (equal) aspect — xy voxels are isotropic, so it must
not be stretched — therefore its plotting box is made SQUARE and the figure height is
computed so that the square image actually fills the 2/3-wide column (instead of being
letter-boxed, which previously made it look ~1/2 wide). The colorbar is drawn in its own
axes so it does not steal width from that square box.

Only the images and their legends are drawn (no matplotlib toolbar, no Qt chrome).
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


DPI = 150
_W = 1800                    # fixed output width in px
_ML, _MR, _MT, _MB = 30, 30, 20, 34   # outer margins (left, right, top, bottom) in px
_GAP = 96                    # gap between depth map and profiles (room for profiles y-labels)
_TITLE_H = 40                # space above the depth map for its title
_CBAR_GAP, _CBAR_H, _CBAR_LABEL = 14, 22, 30   # colorbar gap / strip / tick+label band
_PROF_TITLE = 34             # space above each profile for its title
_PROF_LABEL = 62             # space below each profile for its x tick labels + x label


def export_depth_and_profiles(depth_widget, profiles_widget, filepath, dpi=DPI):
    """
    Renders 'depth_widget' (a DepthMapViewer) and 'profiles_widget' (a ProfilesViewer)
    into a fixed-size figure and saves it as a PNG at 'filepath'.
    """
    # ── horizontal split: 2/3 depth map, 1/3 profiles ───────────────────────
    content_w = _W - _ML - _MR - _GAP
    w_depth = content_w * 2.0 / 3.0
    w_prof = content_w * 1.0 / 3.0
    h_depth = w_depth                       # square plotting box for the depth map

    # ── figure height so the square depth box fits, then everything else ─────
    depth_block = _TITLE_H + h_depth + _CBAR_GAP + _CBAR_H + _CBAR_LABEL
    height = _MT + depth_block + _MB
    facecolor = depth_widget.figure.get_facecolor()
    fig = Figure(figsize=(_W / dpi, height / dpi), dpi=dpi, facecolor=facecolor)
    FigureCanvasAgg(fig)  # a canvas is required for savefig

    # px rect (top-left origin) -> matplotlib figure-fraction [left, bottom, w, h]
    def rect(x, y, w, h):
        return [x / _W, (height - y - h) / height, w / _W, h / height]

    # ── depth map (square) + its colorbar underneath ────────────────────────
    y_depth = _MT + _TITLE_H
    ax_depth = fig.add_axes(rect(_ML, y_depth, w_depth, h_depth))
    ax_cbar = fig.add_axes(rect(_ML, y_depth + h_depth + _CBAR_GAP, w_depth, _CBAR_H))
    depth_widget.render_depth(fig, ax_depth, cax=ax_cbar)
    ax_depth.set_title(depth_widget.title, color=depth_widget.text1_color)

    # ── profiles (yz on top, zx below), right column ────────────────────────
    x_prof = _ML + w_depth + _GAP
    prof_bottom = y_depth + h_depth + _CBAR_GAP + _CBAR_H
    prof_h = ((prof_bottom - _MT) - 2 * (_PROF_TITLE + _PROF_LABEL)) / 2.0
    y_yz = _MT + _PROF_TITLE
    y_zx = y_yz + prof_h + _PROF_LABEL + _PROF_TITLE
    ax_yz = fig.add_axes(rect(x_prof, y_yz, w_prof, prof_h))
    ax_zx = fig.add_axes(rect(x_prof, y_zx, w_prof, prof_h))
    profiles_widget.render_profiles(ax_yz, ax_zx)

    fig.savefig(filepath, dpi=dpi, facecolor=facecolor)
