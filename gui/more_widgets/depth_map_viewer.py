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
        self.figure = Figure(facecolor=qgroupbox_color)
        self.canvas = FigureCanvas(self.figure)
        # the matplotlib toolbar:
        self.toolbar = NavigationToolbar(self.canvas, self)
        ### build the objects together to make the layout:
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def update_plot(self):
        """Show image depth map if image is not None."""
        self.figure.clear()
        self.plot_depths_map()
        self.figure.suptitle(self.title, color=self.text1_color)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_depths_map(self, num_ticks=7):
        """Plots the depth map on the 0-axis ie z-axis (f is 3D image with format ZYX)."""
        ax = self.figure.add_subplot(111)
        image = self.image.clone().cpu().detach().numpy()
        ### plot the depths map:
        nz, ny, nx = image.shape
        colormap = plt.get_cmap(self.cmap)
        map = colormap(np.linspace(0, 1, nz))
        R = np.zeros((ny, nx, nz))
        G = np.zeros((ny, nx, nz))
        B = np.zeros((ny, nx, nz))
        for i in range(nz):
            s = 1.
            slice_i = np.clip(image[i, :, :], None, s)
            R[:, :, i] = slice_i * map[i, 0]
            G[:, :, i] = slice_i * map[i, 1]
            B[:, :, i] = slice_i * map[i, 2]
        proj = np.stack((np.sum(R, axis=2), np.sum(G, axis=2), np.sum(B, axis=2)), axis=2) / nz
        proj = proj / np.max(proj)
        im = ax.imshow(proj)
        ax.axis('off')
        ### add the colorbar:
        norm = Normalize(vmin=self.z0, vmax=self.zN)
        sm = ScalarMappable(cmap=self.cmap, norm=norm)
        cbar = self.figure.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.035, pad=0.05)
        ticks = np.linspace(self.z0, self.zN, num_ticks)
        cbar.set_ticks(ticks, labels=[f"{t:.0f}" for t in ticks])
        cbar.set_label(f'Depth ({self.unit})', fontsize=11, color='#808080')
        cbar.outline.set_color(self.text2_color)
        cbar.ax.tick_params(colors=self.text2_color)
        ### show the depth value in the matplotlib toolbar:
        # calculate the weighted average depth: depth(x, y) = sum_z_(I(z,y,x) * z) / sum_z_(I(z,x,y))
        depths = np.linspace(self.z0, self.zN, nz)
        weights = np.clip(image, 0, 1)
        depth_map = np.tensordot(weights, depths, axes=(0, 0))
        depth_map /= np.sum(weights, axis=0) + 1e-12
        self.depth_map = depth_map  # depth mapped with (x,y) : depth_map.shape == (ny, nx)
        # this attribute is used to show (in the matplolib toolbar) the depth value of the pixel where the mouse is
        # currently on, instead of the RGB value of the pixel:
        im.format_cursor_data = lambda _: ""  # <- don't show the '[R, G, B]'
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

