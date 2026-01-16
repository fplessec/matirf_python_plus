import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistogramWidget(QWidget):
    def __init__(self, image: torch.tensor, parent=None):
        super().__init__()
        self.parent = parent
        image = image.cpu().numpy()
        self.image = image.flatten()  # <- histogram on the entire 3d image
        self.figure = Figure(figsize=(4, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.draw_histogram()

    def draw_histogram(self):
        self.ax.clear()
        self.ax.hist(
            self.image,
            bins=256,
            color='gray',
            edgecolor='black'
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
