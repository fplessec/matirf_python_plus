import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSlider, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import settings


class Image3DViewer(QWidget):

    def __init__(self, image: torch.tensor, parent=None, cmap='gray'):
        super().__init__()
        self.parent = parent
        assert image.ndim == 3, "Image must be 3D (Z,Y,X)"
        self.image = image.detach().cpu()
        self.cmap = cmap
        self.nz, self.h, self.w = self.image.shape
        self.current_slice = 0
        self.current_slice_raw = None
        self.setup_ui()
        self.update_slice()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        ### creation of the widgets one after another:
        # a canvas to render the current slice:
        qgroupbox_color = self.palette().color(QPalette.Mid).name()  # depends on the palette
        self.figure = Figure(facecolor=qgroupbox_color)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)  # no padding or margins
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("figure_leave_event", self.on_mouse_leave)
        # slider to change slice:
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.nz - 1)
        self.slider.setValue(self.current_slice)
        self.slider.valueChanged.connect(self.on_slice_changed)
        # label to show size:
        self.slice_label = QLabel()
        self.size_label = QLabel(f"Size: {self.w} × {self.h} pixels")
        size_label_color = self.palette().color(QPalette.PlaceholderText).name()  # depends on the palette
        self.size_label.setStyleSheet(f"color: {size_label_color}; font-style: italic; font-size: {settings.FontSize.SMALL}pt;")
        # label to show pixel value on mouse:
        self.pixel_label = QLabel("Pixel: (x, y) = -, value = -")
        self.pixel_label.setAlignment(Qt.AlignCenter)
        self.pixel_label.setStyleSheet(f"font-size: {settings.FontSize.SMALL}pt;")
        ### build the objects together to make the layout:
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.slice_label)
        info_layout.addStretch()
        info_layout.addWidget(self.size_label)
        layout.addWidget(self.canvas)
        layout.addLayout(info_layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.pixel_label)
        self.setLayout(layout)

    def on_slice_changed(self, value):
        self.current_slice = value
        self.update_slice()

    def update_slice(self):
        slice_2d = self.image[self.current_slice].numpy()
        self.current_slice_raw = slice_2d  # <- stores the raw values for the mouse
        # normalize (like fiji):
        slice_norm = slice_2d.astype(np.float32)
        slice_norm -= slice_norm.min()
        if slice_norm.max() > 0:
            slice_norm /= slice_norm.max()
        # draw the slice:
        self.ax.clear()
        self.ax.imshow(slice_norm, cmap=self.cmap, origin='upper')
        self.ax.axis('off')  # toujours off
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.draw()
        # update the label:
        self.slice_label.setText(f"Slice {self.current_slice + 1} / {self.nz}")

    def on_mouse_move(self, event):
        """Retrieves the coordinates and values of pixels on the canvas."""
        if self.current_slice_raw is None:
            return
        if event.xdata is None or event.ydata is None:  # <- mouse on the canvas margins
            self.pixel_label.setText("Pixel: (x, y) = -, value = -")
            return
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        if 0 <= x < self.w and 0 <= y < self.h:
            value = self.current_slice_raw[y, x]
            self.pixel_label.setText(f"Pixel: (x={x}, y={y}) → value = {value:.3g}")

    def on_mouse_leave(self, event):
        """Resets the label when the mouse leaves the canvas."""
        self.pixel_label.setText("Pixel: (x, y) = -, value = -")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0 and self.current_slice < self.nz - 1:
            self.slider.setValue(self.current_slice + 1)
        elif delta < 0 and self.current_slice > 0:
            self.slider.setValue(self.current_slice - 1)



if __name__=="__main__":
    import sys
    from in_out import load_tif, MEASUREMENTS_DIR
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("macintosh")  # <- force the style for any OS to macintosh
    window = Image3DViewer(image=load_tif(MEASUREMENTS_DIR / 'esoubies.tif'))
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())