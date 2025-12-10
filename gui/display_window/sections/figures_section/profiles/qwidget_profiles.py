from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from in_out import load_or_create_toml, CONFIG_PATH
from operations import get_variables_from_dict


class Profiles(QWidget):
    def __init__(self):
        super().__init__()
        self.f = None
        self.setup_ui()

    def setup_ui(self):
        """Initialise l'interface du tab Depths Map."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 1, 1, 1)
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
            ax1 = self.figure.add_subplot(211)  # yz
            ax2 = self.figure.add_subplot(212)  # zx
            self.plot_profiles(ax1, ax2, self.f, z0, zN)

            self.figure.suptitle("Profiles")
            self.figure.tight_layout()
            self.canvas.draw()

    def plot_profiles(self, ax1, ax2, f, z0, zN, num_ticks=5):
        f_ = f.clone().cpu().detach()
        z_ticks_label = np.linspace(z0, zN, num_ticks)
        z_ticks = np.linspace(0, self.f.shape[0] - 1, num_ticks)

        ax1.set_xlabel("nm")
        ax1.set_ylabel("pix")
        ax1.set_xticks(z_ticks, labels=[f"{z:.0f}" for z in z_ticks_label])
        ax2.set_xlabel("pix")
        ax2.set_ylabel("nm")
        ax2.set_yticks(z_ticks, labels=[f"{z:.0f}" for z in z_ticks_label])

        vmin = min(f_.mean(dim=2).min(), f_.mean(dim=1).min(), f_.mean(dim=0).min()).item()
        vmax = max(f_.mean(dim=2).max(), f_.mean(dim=1).max(), f_.mean(dim=0).max()).item()

        _ = ax1.imshow(f_.mean(dim=2).transpose(0, 1), aspect='auto', vmin=vmin, vmax=vmax)
        ax1.set_title("yz")
        ax2.imshow(f_.mean(dim=1), aspect='auto', vmin=vmin, vmax=vmax)
        ax2.set_title("zx")
        # ax4 = self.figure.add_subplot(2,20,1)
        # self.figure.colorbar(_, cax=ax4, orientation='vertical')