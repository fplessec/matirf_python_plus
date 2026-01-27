import torch
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

import settings


class ProfilesViewer(QWidget):
    """Viewer for 3D profile data with two projections: yz and zx."""

    def __init__(self, image, z0, zN, unit='nm', title='Profiles', cmap='viridis'):
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
        layout.setContentsMargins(1, 1, 1, 1)
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
        """Update the plot based on current data."""
        self.figure.clear()
        self.plot_profiles()
        self.figure.suptitle(self.title, color=self.text1_color)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_profiles(self, num_ticks=5):
        ax1 = self.figure.add_subplot(211)  # yz
        ax2 = self.figure.add_subplot(212)  # zx
        image = self.image.clone().cpu().detach()
        # set up the labels and ticks of the two axis of the figure:
        ax1.set_xlabel(self.unit, color=self.text2_color)
        ax1.set_ylabel("pix", color=self.text2_color)
        ax2.set_xlabel("pix", color=self.text2_color)
        ax2.set_ylabel(self.unit, color=self.text2_color)
        z_ticks_label = np.linspace(self.z0, self.zN, num_ticks)
        z_ticks = np.linspace(0, image.shape[0] - 1, num_ticks)
        y_pix_tick = np.linspace(0, image.shape[1] - 1, num_ticks)
        x_pix_tick = np.linspace(0, image.shape[2] - 1, num_ticks)
        ax1.set_xticks(z_ticks, labels=[f"{z:.0f}" for z in z_ticks_label], color=self.text2_color)
        ax1.set_yticks(y_pix_tick, labels=[int(round(y)) for y in y_pix_tick+1], color=self.text2_color)
        ax2.set_yticks(z_ticks, labels=[f"{z:.0f}" for z in z_ticks_label], color=self.text2_color)
        ax2.set_xticks(x_pix_tick, labels=[int(round(x)) for x in x_pix_tick+1], color=self.text2_color)
        ax1.tick_params(colors=self.text2_color)
        ax2.tick_params(colors=self.text2_color)
        for spine in ax1.spines.values(): spine.set_color(self.text2_color)
        for spine in ax2.spines.values(): spine.set_color(self.text2_color)
        # compute min/max for consistent color scaling
        vmin = min(image.mean(dim=2).min(), image.mean(dim=1).min()).item()
        vmax = max(image.mean(dim=2).max(), image.mean(dim=1).max()).item()
        # plot the profiles: two projections of the 3D data: along yz and zx
        ax1.imshow(image.mean(dim=2).transpose(0, 1), aspect='auto', vmin=vmin, vmax=vmax, cmap=self.cmap)
        ax1.set_title("yz", color=self.text1_color)
        ax2.imshow(image.mean(dim=1), aspect='auto', vmin=vmin, vmax=vmax, cmap=self.cmap)
        ax2.set_title("zx", color=self.text1_color)


if __name__=="__main__":
    import sys
    from in_out import load_tif, RESULTS_DIR
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    f = load_tif(RESULTS_DIR / 'ADMM'/ 'f.tif')
    window = ProfilesViewer(image=f, z0=0, zN=300)
    window.resize(450, 900)
    window.show()
    sys.exit(app.exec_())