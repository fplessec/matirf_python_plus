import torch
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import settings


class Image2DViewer(QWidget):
    """
    An object that allows interactive visualization of a 2D image (Y,X):
        > displays the image with imshow
        > shows pixel intensity under the mouse cursor in real time
        > provides a callback:
            - mouse_moved_callback(x, y): triggered when the mouse moves over a valid pixel
        > adapts dynamically when set_image() is called with a new tensor
    """
    def __init__(self, image: torch.Tensor, parent=None, cmap='gray'):
        super().__init__()
        self.parent = parent
        self.cmap = cmap
        self.image_raw = None
        self.h = 0
        self.w = 0
        self.current_x = None
        self.current_y = None
        self.mouse_moved_callback = None
        self._setup_ui()
        self.set_image(image)

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        ## canvas to render the image:
        qgroupbox_color = self.palette().color(QPalette.Mid).name()
        self.figure = Figure(facecolor=qgroupbox_color)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("figure_leave_event", self._on_mouse_leave)
        ## info labels:
        self.size_label = QLabel()
        size_label_color = self.palette().color(QPalette.PlaceholderText).name()
        self.size_label.setStyleSheet(
            f"color: {size_label_color}; font-style: italic; font-size: {settings.FontSize.SMALL}pt;")
        self.pixel_label = QLabel("Pixel: (x, y) = -, value = -")
        self.pixel_label.setAlignment(Qt.AlignCenter)
        self.pixel_label.setStyleSheet(f"font-size: {settings.FontSize.SMALL}pt;")
        ## assemble:
        info_layout = QHBoxLayout()
        info_layout.addStretch()
        info_layout.addWidget(self.size_label)
        layout.addWidget(self.canvas)
        layout.addLayout(info_layout)
        layout.addWidget(self.pixel_label)
        self.setLayout(layout)

    ## replaces the displayed image with a new tensor:
    def set_image(self, image: torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        assert image.ndim == 2, f"Image must be 2D (Y,X), got shape {tuple(image.shape)}"
        self.image_raw = image.detach().cpu().numpy()
        self.h, self.w = self.image_raw.shape
        self.size_label.setText(f"Size: {self.w} x {self.h} pixels")
        self._redraw()

    def _redraw(self):
        img = self.image_raw.astype(np.float32)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        self.ax.clear()
        self.ax.imshow(img, cmap=self.cmap, origin='upper')
        self.ax.axis('off')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.draw()

    def _on_mouse_move(self, event):
        if self.image_raw is None:
            return
        if event.xdata is None or event.ydata is None:
            self.pixel_label.setText("Pixel: (x, y) = -, value = -")
            self.current_x = None
            self.current_y = None
            if self.mouse_moved_callback is not None:
                self.mouse_moved_callback(None, None)
            return
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        if not (0 <= x < self.w and 0 <= y < self.h):
            self.pixel_label.setText("Pixel: (x, y) = -, value = -")
            if self.mouse_moved_callback is not None:
                self.mouse_moved_callback(None, None)
            return
        self.current_x = x
        self.current_y = y
        value = self.image_raw[y, x]
        self.pixel_label.setText(f"Pixel: (x={x}, y={y}) -> value = {value:.3g}")
        if self.mouse_moved_callback is not None:
            self.mouse_moved_callback(x, y)

    def _on_mouse_leave(self, event):
        self.pixel_label.setText("Pixel: (x, y) = -, value = -")
        self.current_x = None
        self.current_y = None
        if self.mouse_moved_callback is not None:
            self.mouse_moved_callback(None, None)
