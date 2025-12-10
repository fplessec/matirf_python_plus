from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np

from in_out import load_or_create_toml, CONFIG_PATH
from operations import get_variables_from_dict


class DepthMap(QWidget):
    def __init__(self):
        super().__init__()
        self.f = None
        self.setup_ui()

    def setup_ui(self):
        """Initialise l'interface du tab Depths Map."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(1, 1, 0, 1)
        self.layout.setSpacing(0)

        # Crée une figure matplotlib et un canevas pour l'affichage
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        # Label pour afficher un message si f est None
        self.empty_label = QLabel()

        # Ajoute les widgets dans le layout
        self.layout.addWidget(self.empty_label)
        self.layout.addWidget(self.canvas)

        # Ajoute la barre d'outils Matplotlib en bas
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        # Cache les éléments au début si nécessaire
        self.empty_label.hide()
        self.update_plot()

    def update_plot(self):
        """Vérifie si f existe, sinon affiche un message."""
        if self.f is None:
            self.canvas.hide()
            self.empty_label.show()
            self.toolbar.hide()
        else:
            self.empty_label.hide()
            self.canvas.show()
            self.toolbar.show()

            config = load_or_create_toml(CONFIG_PATH)
            (z0, zN) = get_variables_from_dict(config['oper-params'], ['z0', 'zN'])
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            self.plot_depths_map(ax, self.f, z0, zN)

            self.figure.suptitle("Depth map")
            self.figure.tight_layout()
            self.canvas.draw()

    def plot_depths_map(self, ax, f, z0, zN, cmap='jet_r', unit='nm'):
        """Affiche la depth map sur l'axe 0 (f est une image 3D format ZYX)."""
        f_ = f.clone().cpu().detach()
        nz, ny, nx = f_.shape
        colormap = plt.get_cmap(cmap)
        map = colormap(np.linspace(0, 1, nz))
        R = np.zeros((ny, nx, nz))
        G = np.zeros((ny, nx, nz))
        B = np.zeros((ny, nx, nz))
        for i in range(nz):
            s = 1.
            R[:, :, i] = f_[i, :, :].clamp(max=s) * map[i, 0]
            G[:, :, i] = f_[i, :, :].clamp(max=s) * map[i, 1]
            B[:, :, i] = f_[i, :, :].clamp(max=s) * map[i, 2]
        proj = np.stack((np.sum(R, axis=2), np.sum(G, axis=2), np.sum(B, axis=2)), axis=2) / nz
        proj = proj / np.max(proj)

        ax.imshow(proj)
        ax.axis('off')

        norm = Normalize(vmin=z0, vmax=zN)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = self.figure.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.035, pad=0.05)
        cbar.set_label(f'Depth ({unit})', fontsize=11)
