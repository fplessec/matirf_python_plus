from os import makedirs
from os.path import join

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QStackedWidget
from tomli_w import dumps

from algorithms import ALGORITHMS
from gui.display_window.sections.figures_section import FiguresSection
from gui.display_window.sections.message_section import MessageSection
import settings
from in_out import load_tif, load_json, RESULTS_DIR, save_tif, save_toml, save_txt
from preprocess_measurement import preprocess_measurement_stack
from operations import compute_matirf_operator_from_params, apply_matirf_operator
from .synthetic_truth_layout import SyntheticTruthSection


class DisplayWindow(QMainWindow):
    def __init__(self, config, f=None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.f = f
        self.config = config
        self.setWindowTitle("Display Window")
        self.resize(settings.width_dw, settings.height_dw)
        self.setup_ui()
        self.setup_reconstruction()

    def setup_ui(self):
        self.menuBar()
        # the base objects:
        central_widget = QWidget()
        main_layout = QHBoxLayout()  # Layout horizontal principal pour les deux colonnes
        # create the specifics layouts/sections one after another:
        right_column = self.create_right_column()
        left_column = self.create_left_column()
        # build the objects together to make the window:
        main_layout.addLayout(left_column, 2)  # 2/7ieme de la largeur
        main_layout.addLayout(right_column, 5)  #5/7ieme tiers de la largeur
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_left_column(self):
        left_layout = QVBoxLayout()
        # Section config (1 tier de la hauteur)
        config_section = MessageSection("Config")
        config_section._print(dumps(self.config))
        # Section messages (2 tiers de la hauteur)
        self.messages_section = MessageSection("Messages")
        self.bottom_bar = self.create_bottom_bar()
        # Ajouter les widgets avec les proportions souhaitées
        left_layout.addWidget(config_section, 1)  # 1 tier de la hauteur
        left_layout.addWidget(self.messages_section, 2)  # 2 tiers de la hauteur
        left_layout.addWidget(self.bottom_bar)
        return left_layout

    def create_right_column(self):
        right_layout = QVBoxLayout()
        # right column is a QStackedWidget and can switch between some sections:
        self.right_column_stack = QStackedWidget()
        self.figures_section = FiguresSection()
        self.right_column_stack.addWidget(self.figures_section)  # index = 0
        if self._is_synthetic_data():
            self.synthetic_truth_section = SyntheticTruthSection()
            self.right_column_stack.addWidget(self.synthetic_truth_section)  # index = 1
        right_layout.addWidget(self.right_column_stack)
        return right_layout

    def create_bottom_bar(self):
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        self.save_recons_button = QPushButton("Save reconstruction")
        self.save_recons_button.setEnabled(self._is_button_enable())
        self.save_recons_button.clicked.connect(self.save_reconstruction)
        bottom_layout.addWidget(self.save_recons_button)
        bottom_layout.addStretch()
        if self._is_synthetic_data():
            self.change_right_column_stack_button = QPushButton("Go To Truth →")
            self.change_right_column_stack_button.clicked.connect(self.change_right_column)
            self.change_right_column_stack_button.setEnabled(self._is_button_enable())
            bottom_layout.addWidget(self.change_right_column_stack_button)
        bottom_widget.setLayout(bottom_layout)
        return bottom_widget

    def _is_button_enable(self):
        return self.f is not None

    def _is_synthetic_data(self):
        return self.config['input-paths']['mode'] == 'synthetic-data'

    def setup_reconstruction(self):
        if self.f is None:
            if not self._is_synthetic_data():
                tif_path = self.config['input-paths']['tif']
                json_path = self.config['input-paths']['json']
                g = load_tif(tif_path)
                measurement_params = load_json(json_path)
                add_noise_params = self.config['add-noise']
                self.g, measurement_params = preprocess_measurement_stack(g, measurement_params, add_noise_params)
                oper_params = self.config['oper-params']
                self.H = compute_matirf_operator_from_params(measurement_params, oper_params)
                self.algo_params = self.config['algo-params']
                self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"]()
            else:  # synthetic data
                tif_path = self.config['input-paths']['tif']
                json_path = self.config['input-paths']['json']
                self.f_true = load_tif(tif_path)
                measurement_params = load_json(json_path)
                oper_params = self.config['oper-params']
                nz_true = self.f_true.shape[0]
                nz = oper_params['nz']
                assert nz == nz_true, (
                    f"\nWhile computing the synthetic MA-TIRF measurement:\nThe parameter 'nz' in 'oper-params' (nz = "
                    f"{nz}) must be equal to the number of plans in the synthetic truth ({nz_true})."
                )
                self.H = compute_matirf_operator_from_params(measurement_params, oper_params)
                g = apply_matirf_operator(self.H, self.f_true)
                add_noise_params = self.config['add-noise']
                self.g, _ = preprocess_measurement_stack(g, measurement_params, add_noise_params)
                self.algo_params = self.config['algo-params']
                self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"]()
        else:  # no need to setup the reconstruction: just need to update the f in the figures
               # ie happens when user clicked on 'Open reconstruction'
            self.update_f(self.f)
            self.update_plot()

    def update_f(self, f):
        self.figures_section.depth_map_widget.f = f
        self.figures_section.profiles_widget.f = f
        self.f = f

    def update_plot(self):
        self.figures_section.depth_map_widget.update_plot()
        self.figures_section.profiles_widget.update_plot()
        self.save_recons_button.setEnabled(self._is_button_enable())
        if self._is_synthetic_data():
            self.change_right_column_stack_button.setEnabled(self._is_button_enable())

    def save_reconstruction(self):
        """Ouvre un dialogue pour choisir un dossier et enregistre la reconstruction."""
        save_dir, _ = QFileDialog.getSaveFileName(self, "Select Save Directory", str(RESULTS_DIR),
                                                  "Folder Selection (*.*)")
        if save_dir not in ['', RESULTS_DIR, None]:
            makedirs(save_dir)
            save_tif(self.f, join(save_dir, 'f.TIF'))
            save_toml(self.config, join(save_dir, 'config.toml'))
            messages_section_content = self.messages_section.get_text()
            save_txt(messages_section_content, join(save_dir, 'messages.txt'))
            if self._is_synthetic_data():
                save_tif(self.f_true, join(save_dir, 'f_true.TIF'))
            self._print(f"Saved reconstruction in {save_dir}")

    def run(self):
        if hasattr(self, 'algorithm'):
            print("Running started.\n")
            self.f = self.algorithm._run(self.g, self.H, self.algo_params, window=self)

    def change_right_column(self):
        if self.right_column_stack.currentIndex() == 0:
            self.right_column_stack.setCurrentIndex(1)
            self.change_right_column_stack_button.setText("← Go to Reconstruction")
        else:
            self.right_column_stack.setCurrentIndex(0)
            self.change_right_column_stack_button.setText("Go To Truth →")

    @pyqtSlot(str)
    def _print(self, string):
        if hasattr(self, 'messages_section'):
            self.messages_section._print(string)
        else:
            print(string)

    def closeEvent(self, event):
        if hasattr(self, 'algorithm') and self.algorithm:
            self.algorithm.stop_running()
        from ..control_window import DisplayWindowManager
        DisplayWindowManager.remove(self)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # escape is clicked
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        elif event.key() == Qt.Key_W and event.modifiers() & Qt.ControlModifier:  # Ctrl+W is clicked
            self.close()
        elif event.key() == Qt.Key_A:
            print("A pressed")
            self.update_plot()
        else:
            super().keyPressEvent(event)