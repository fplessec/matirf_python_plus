import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistogramWidget(QWidget):
    def __init__(self, image: torch.tensor, bins=256, parent=None):
        super().__init__()
        self.parent = parent
        image = image.cpu().detach().numpy()
        self.image = image.flatten()  # <- histogram on the entire 3d image
        self.bins = bins
        self.setup_ui()
        self.draw_histogram()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(4, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def draw_histogram(self):
        self.ax.clear()
        self.ax.hist(
            self.image,
            bins=self.bins,
            color='gray',
            edgecolor='gray'
        )
        self.ax.set_yscale("log")
        self.ax.set_title("Histogram of the entire 3D object:")
        self.ax.set_xlabel("Pixel value")
        self.ax.set_ylabel("Count (log)")
        self.figure.tight_layout()
        self.canvas.draw_idle()


if __name__=="__main__":
    import sys
    from in_out import load_tif, MEASUREMENTS_DIR
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = HistogramWidget(image=load_tif(MEASUREMENTS_DIR / 'esoubies.tif'))
    window.show()
    sys.exit(app.exec_())
