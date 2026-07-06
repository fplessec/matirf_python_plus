import torch
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Histogram2DWidget(QWidget):
    """
    An object that can render the histogram of a 2D image in 2 different modes:
        > mode='Full': shows the histogram of the entire 2D image
        > mode='XY-patch': shows the histogram of a local patch of size (2p+1)x(2p+1)
          centered at a given (x, y) position
    """
    def __init__(self, image: torch.Tensor, bins=256, mode='Full', scale='log', patch_radius=2, parent=None):
        super().__init__()
        self.parent = parent
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        self.image_2d = image.cpu().detach().numpy()
        self.bins = bins
        self.mode = mode  # 'Full' or 'XY-patch'
        self.scale = scale  # 'log' or 'linear'
        self.current_x = None
        self.current_y = None
        self.patch_radius = patch_radius
        ## precomputed histogram for the full image:
        self.hist_range = (self.image_2d.min(), self.image_2d.max())
        self.hist_full, self.hist_bins = np.histogram(
            self.image_2d, bins=self.bins, range=self.hist_range)
        self.bar_container = None
        self._setup_ui()
        self._update_histogram()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(4, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def set_mode(self, mode: str):
        if mode not in ['Full', 'XY-patch']:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.bar_container = None
        self._update_histogram()

    def set_scale(self, scale: str):
        if scale not in ['log', 'linear']:
            raise ValueError(f"Unknown scale: {scale}")
        self.scale = scale
        self._update_histogram()

    def set_xy(self, x: int, y: int):
        self.current_x = x
        self.current_y = y
        if self.mode == 'XY-patch':
            self._update_histogram()

    ## replaces the image data and recomputes histogram:
    def set_image(self, image: torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        self.image_2d = image.cpu().detach().numpy()
        self.hist_range = (self.image_2d.min(), self.image_2d.max())
        self.hist_full, self.hist_bins = np.histogram(
            self.image_2d, bins=self.bins, range=self.hist_range)
        self.bar_container = None
        self._update_histogram()

    def _get_patch_data(self):
        if self.current_x is None or self.current_y is None:
            return None
        h, w = self.image_2d.shape
        x = int(np.clip(self.current_x, 0, w - 1))
        y = int(np.clip(self.current_y, 0, h - 1))
        p = self.patch_radius
        x0, x1 = max(0, x - p), min(w, x + p + 1)
        y0, y1 = max(0, y - p), min(h, y + p + 1)
        return self.image_2d[y0:y1, x0:x1].reshape(-1)

    def _init_bar_plot(self, hist):
        bins = self.hist_bins
        self.bar_container = self.ax.bar(
            bins[:-1], hist, width=np.diff(bins),
            align='edge', color='gray', edgecolor='gray')

    def _update_histogram(self):
        hist = None
        if self.mode == 'Full':
            hist = self.hist_full
        elif self.mode == 'XY-patch':
            data = self._get_patch_data()
            if data is not None:
                hist, _ = np.histogram(data, bins=self.hist_bins)
        # update bars:
        if hist is not None:
            if self.bar_container is None:
                self.ax.clear()
                self._init_bar_plot(hist)
            else:
                for rect, h in zip(self.bar_container, hist):
                    rect.set_height(h)
        else:
            self.ax.clear()
            self.bar_container = None
        # scale:
        if self.scale == 'log':
            self.ax.set_yscale("log", nonpositive='clip')
        else:
            self.ax.set_yscale("linear")
        # titles:
        if self.mode == 'Full':
            self.ax.set_title("Histogram of the entire image:")
        elif self.mode == 'XY-patch':
            if self.current_x is None:
                self.ax.set_title("Histogram of local patch: (no pixel selected)")
            else:
                size = 2 * self.patch_radius + 1
                self.ax.set_title(
                    f"Histogram of patch ({size}x{size}) at (x={self.current_x}, y={self.current_y})")
        self.ax.set_xlabel("Pixel value")
        self.ax.set_ylabel("Count")
        self.figure.tight_layout()
        self.canvas.draw_idle()
