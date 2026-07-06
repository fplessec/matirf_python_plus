import torch
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Histogram3DWidget(QWidget):
    """
    An object that can render the histogram of a 3D image in 3 different manner:
        > mode='3D': shows the histogram of the entire 3D volume
        > mode='Z-Slice': shows the histogram of a given 2D Z-slice
        > mode='XY-depth-column': shows the histogram of all pixel intensities
          within a local 3D column (spatial patch of size (2p+1)x(2p+1) extended along Z, with p=patch_radius)
          centered at a given (x, y)
    """
    def __init__(self, image: torch.Tensor, bins=256, mode='3D', scale='log', patch_radius=2, parent=None):
        super().__init__()
        self.parent = parent
        image = image.cpu().detach().numpy()
        self.image_3d = image  # <- full 3d image
        self.bins = bins
        self.mode = mode  # '3D', 'Z-slice', or 'XY-depth-column'
        self.scale = scale  # <- 'log' or 'linear'
        self.current_slice = 0
        self.current_x = None
        self.current_y = None
        self.patch_radius = patch_radius  # <- radius p (patch size = 2p+1)
        # attributes in order to store Z-slice histograms and their parameters to gain fluidity
        # if current_slide changes rapidly:
        self.precomputed_hist = None
        self.hist_bins = None
        self.hist_range = None
        self.hist_3d = None
        self.bar_container = None
        self.precompute_z_histograms()
        self.setup_ui()
        self.update_histogram()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(4, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    ## replaces the image data and recomputes all histograms:
    def set_image(self, image):
        self.image_3d = image.cpu().detach().numpy()
        self.bar_container = None
        self.precompute_z_histograms()
        self.update_histogram()

    def set_mode(self, mode: str):
        """Change histogram mode."""
        if mode not in ['3D', 'Z-slice', 'XY-depth-column']:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.bar_container = None  # <- reset bars
        self.update_histogram()

    def set_scale(self, scale: str):
        """Change y-axis scale ('log' or 'linear')."""
        if scale not in ['log', 'linear']:
            raise ValueError(f"Unknown scale: {scale}")
        self.scale = scale
        self.update_histogram()

    def set_slice(self, slice_index: int):
        """Update current slice (used in Z-slice mode)."""
        self.current_slice = slice_index
        if self.mode == 'Z-slice':
            self.update_histogram()

    def set_xy(self, x: int, y: int):
        """Update (x,y) for XY-depth-column mode."""
        self.current_x = x
        self.current_y = y
        if self.mode == 'XY-depth-column':
            self.update_histogram()

    def get_data(self):
        """Return data depending on the current mode."""
        if self.mode == 'XY-depth-column':
            if self.current_x is None or self.current_y is None:
                return None
            x = int(np.clip(self.current_x, 0, self.image_3d.shape[2] - 1))
            y = int(np.clip(self.current_y, 0, self.image_3d.shape[1] - 1))
            p = self.patch_radius
            # bounds of the patch
            x0 = max(0, x - p)
            x1 = min(self.image_3d.shape[2], x + p + 1)
            y0 = max(0, y - p)
            y1 = min(self.image_3d.shape[1], y + p + 1)
            # extract full patch volume
            patch = self.image_3d[:, y0:y1, x0:x1]  # shape: (nz, dy, dx)
            # flatten EVERYTHING → all pixels in the column volume
            return patch.reshape(-1)
        return None

    def precompute_z_histograms(self):
        """Precompute histograms for all Z slices and 3D volume."""
        nz = self.image_3d.shape[0]
        vmin = self.image_3d.min()
        vmax = self.image_3d.max()
        self.hist_range = (vmin, vmax)
        _, bin_edges = np.histogram(self.image_3d[0], bins=self.bins, range=self.hist_range)
        self.hist_bins = bin_edges
        self.precomputed_hist = np.zeros((nz, self.bins), dtype=np.float32)
        for z in range(nz):
            hist, _ = np.histogram(self.image_3d[z], bins=self.bins, range=self.hist_range)
            self.precomputed_hist[z] = hist
        self.hist_3d, _ = np.histogram(self.image_3d, bins=self.bins, range=self.hist_range)

    def init_bar_plot(self, hist):
        bins = self.hist_bins
        self.bar_container = self.ax.bar(bins[:-1], hist, width=np.diff(bins),
                                         align='edge', color='gray', edgecolor='gray')

    def update_histogram(self):
        hist = None
        if self.mode == 'Z-slice':
            z = np.clip(self.current_slice, 0, self.image_3d.shape[0] - 1)
            hist = self.precomputed_hist[z]  # <- already precomputed
        elif self.mode == '3D':
            hist = self.hist_3d  # <- already precomputed
        elif self.mode == 'XY-depth-column':  # <- not precomputed, but way fewer data so way faster to compute
            data = self.get_data()
            if data is not None:
                hist, _ = np.histogram(data, bins=self.hist_bins)
        # update bars:
        if hist is not None:
            if self.bar_container is None:
                self.ax.clear()
                self.init_bar_plot(hist)
            else:
                for rect, h in zip(self.bar_container, hist):
                    rect.set_height(h)
        else:
            self.ax.clear()
            self.bar_container = None
        # and scale:
        if self.scale == 'log':
            self.ax.set_yscale("log", nonpositive='clip')
        else:
            self.ax.set_yscale("linear")
        # and titles:
        if self.mode == '3D':
            self.ax.set_title("Histogram of the entire 3D object:")
        elif self.mode == 'Z-slice':
            self.ax.set_title(f"Histogram of Z slice {self.current_slice+1}:")
        elif self.mode == 'XY-depth-column':
            if self.current_x is None:
                self.ax.set_title("Histogram of depth column: (no pixel selected)")
            else:
                size = 2 * self.patch_radius + 1
                self.ax.set_title(
                    f"Histogram of column volume ({size}x{size}xZ) at (x={self.current_x}, y={self.current_y})"
                )
        # and the rest:
        self.ax.set_xlabel("Pixel value")
        self.ax.set_ylabel("Count")
        self.figure.tight_layout()
        self.canvas.draw_idle()



if __name__=="__main__":  # test
    import sys
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory, QPushButton, QGroupBox

    import gui.more_widgets as more_widgets
    from in_out import load_tif
    import settings


    class HistogramTestWidget(QGroupBox):
        def __init__(self, image, title='title'):
            super().__init__(title=title)
            self.image = image
            self.current_slice = 0
            self.histogram = Histogram3DWidget(image=image, mode='3D')
            self.setup_ui()
        def setup_ui(self):
            layout = QVBoxLayout()
            layout.addWidget(self.histogram)
            # button: swap mode
            self.btn_mode = QPushButton("swap mode")
            self.btn_mode.clicked.connect(self.swap_mode)
            layout.addWidget(self.btn_mode)
            # button: next slice
            self.btn_next = QPushButton("next slice")
            self.btn_next.clicked.connect(self.next_slice)
            layout.addWidget(self.btn_next)
            # button: pick pixel
            self.btn_pick = QPushButton("pick pixel")
            self.btn_pick.clicked.connect(self.pick_pixel)
            layout.addWidget(self.btn_pick)
            self.setLayout(layout)
        def swap_mode(self):
            modes = ['3D', 'Z-slice', 'XY-depth-column']
            current = self.histogram.mode
            next_mode = modes[(modes.index(current) + 1) % len(modes)]
            self.histogram.set_mode(next_mode)
            # scale logic
            if next_mode == 'XY-depth-column':
                self.histogram.set_scale('linear')
            else:
                self.histogram.set_scale('log')
        def next_slice(self):
            if self.histogram.mode != 'Z-slice':
                return
            self.current_slice += 1
            if self.current_slice >= self.image.shape[0]:
                self.current_slice = 0
            self.histogram.set_slice(self.current_slice)
        def pick_pixel(self):
            if self.histogram.mode != 'XY-depth-column':
                return
            ny, nx = self.image.shape[1], self.image.shape[2]
            x = np.random.randint(0, nx)
            y = np.random.randint(0, ny)
            self.histogram.set_xy(x, y)


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette

    package_path = Path(more_widgets.__file__).parent
    image3d = load_tif(package_path / "_image_for_test.TIF")

    window = HistogramTestWidget(image=image3d, title='test of object: Histogram3DWidget')
    window.resize(600, 680)
    window.show()

    sys.exit(app.exec_())