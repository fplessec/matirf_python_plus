from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np


class DepthMapViewer(QWidget):
    """
    An object that renders a 2D depth map projection from a 3D image:
        > computes a color-coded projection along the Z axis using a colormap
        > displays a horizontal colorbar mapping depth values (z0 → zN)
        > shows the estimated depth at each (x, y) position using a weighted average
        > displays depth values interactively in the matplotlib toolbar when hovering the image
    """
    def __init__(self, image, z0, zN, unit='nm', title='Depths map', cmap='jet_r'):
        super().__init__()
        self.image = image
        self.z0 = z0
        self.zN = zN
        self.unit = unit
        self.cmap = cmap
        self.title = title
        self.text1_color = self.palette().color(QPalette.WindowText).name()  # depends on the palette
        self.text2_color = self.palette().color(QPalette.PlaceholderText).name()  # depends on the palette
        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 0, 1)
        layout.setSpacing(0)
        ### creation of the widgets one after another:
        # a canvas to render the depth map:
        qgroupbox_color = self.palette().color(QPalette.Mid).name()  # depends on the palette
        # constrained layout keeps the image + colorbar within the widget bounds and
        # re-flows on resize (no overflow onto the neighbouring profiles viewer):
        self.figure = Figure(facecolor=qgroupbox_color, layout='constrained')
        self.canvas = FigureCanvas(self.figure)
        # the matplotlib toolbar:
        self.toolbar = NavigationToolbar(self.canvas, self)
        ### build the objects together to make the layout:
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    ## replaces the image data and redraws the depth map:
    def set_image(self, image):
        self.image = image
        ## fast path: update existing AxesImage data without clearing the figure:
        if hasattr(self, '_imshow_obj') and self._imshow_obj is not None:
            proj, depth_map = self._compute_projection()
            self._imshow_obj.set_data(proj)
            self.depth_map = depth_map
            self.canvas.draw_idle()
            return
        ## full redraw (initial call):
        self.update_plot()

    def update_plot(self):
        """Show image depth map if image is not None."""
        self._imshow_obj = None
        self.figure.clear()
        self.plot_depths_map()
        self.figure.suptitle(self.title, color=self.text1_color)
        self.canvas.draw()

    def _compute_projection(self):
        """Computes the RGB depth projection and the weighted-average depth map (vectorized)."""
        image = self.image.clone().cpu().detach().numpy()
        nz, ny, nx = image.shape
        colormap = plt.get_cmap(self.cmap)
        colors = colormap(np.linspace(0, 1, nz))[:, :3]  # (nz, 3)
        ## clamp and weight each slice by its colormap color (fully vectorized):
        clamped = np.clip(image, None, 1.0)                # (nz, ny, nx)
        proj = np.einsum('zyx,zc->yxc', clamped, colors)   # (ny, nx, 3)
        proj /= nz
        proj /= np.max(proj) + 1e-12
        np.clip(proj, 0.0, 1.0, out=proj)
        ## weighted average depth:
        depths = np.linspace(self.z0, self.zN, nz)
        weights = np.clip(image, 0, 1)
        depth_map = np.tensordot(weights, depths, axes=(0, 0))
        depth_map /= np.sum(weights, axis=0) + 1e-12
        return proj, depth_map

    def render_depth(self, fig, ax, num_ticks=7, cax=None):
        """
        Draws the depth-map projection + horizontal colorbar onto the given (fig, ax).

        Reused both by the interactive viewer and by the fixed-size PNG export, so the
        exported figure is visually identical to what is shown on screen.
        When 'cax' is given, the colorbar is drawn in that dedicated axes instead of
        stealing space from 'ax' (this lets the export keep 'ax' exactly square).
        Returns (imshow_obj, depth_map).
        """
        proj, depth_map = self._compute_projection()
        imshow_obj = ax.imshow(proj)
        ax.axis('off')
        ### add the colorbar:
        norm = Normalize(vmin=self.z0, vmax=self.zN)
        sm = ScalarMappable(cmap=self.cmap, norm=norm)
        if cax is not None:
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        else:
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.035, pad=0.05)
        ticks = np.linspace(self.z0, self.zN, num_ticks)
        cbar.set_ticks(ticks, labels=[f"{t:.0f}" for t in ticks])
        cbar.set_label(f'Depth ({self.unit})', fontsize=11, color='#808080')
        cbar.outline.set_color(self.text2_color)
        cbar.ax.tick_params(colors=self.text2_color)
        return imshow_obj, depth_map

    def plot_depths_map(self, num_ticks=7):
        """Plots the depth map on the 0-axis ie z-axis (f is 3D image with format ZYX)."""
        ax = self.figure.add_subplot(111)
        nz, ny, nx = self.image.shape
        self._imshow_obj, self.depth_map = self.render_depth(self.figure, ax, num_ticks)
        ### show the depth value in the matplotlib toolbar:
        # this attribute is used to show (in the matplolib toolbar) the depth value of the pixel where the mouse is
        # currently on, instead of the RGB value of the pixel:
        self._imshow_obj.format_cursor_data = lambda _: ""  # <- don't show the '[R, G, B]'
        def format_coord(x, y):  # overwrite the format_coord function of the ax object (mouse event related)
            ix, iy = int(x), int(y)
            if 0 <= ix < nx and 0 <= iy < ny:
                d = self.depth_map[iy, ix]
                return f"x={ix}, y={iy}\ndepth={d:.2f} {self.unit}"
            return ""
        ax.format_coord = format_coord


if __name__=="__main__":  # test
    import sys
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory, QGroupBox

    import gui.more_widgets as more_widgets
    from in_out import load_tif
    import settings


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette

    package_path = Path(more_widgets.__file__).parent
    image3d = load_tif(package_path / "_image_for_test.TIF")

    window = QGroupBox(title='test of object: ImageAndHisto3DViewer')
    layout = QVBoxLayout()
    layout.addWidget(DepthMapViewer(image=image3d, z0=0, zN=300))
    window.setLayout(layout)
    window.resize(600, 600)
    window.show()

    sys.exit(app.exec_())

