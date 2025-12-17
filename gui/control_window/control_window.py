from os.path import join

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QAction

import settings
from in_out import load_or_create_toml, CONFIG_PATH, RESULTS_DIR, save_toml, load_tif, load_txt
from .display_window_manager import DisplayWindowManager
from .sections import InputFilesSection, OperatorParametersSection, AlgorithmAndAlgoParamsSection, AddNoiseSection
from ..display_window import DisplayWindow


class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MA-TIRF Parameter Selection")
        self.setGeometry(100, 100, settings.width_cw, settings.height_cw)
        self.setup_ui()
        # if it's the first time the config file it's create, allow it no fill the necessary parameters to 'None',
        # and to put the bool parameters to their default value:
        self.load_cached_config()

    def setup_ui(self):
        self.setFocus()
        self.create_menu_bar()
        # the base objects:
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        columns_layout = QHBoxLayout()
        # create the specifics layouts one after another:
        left_column = self.create_left_layout()
        right_column = self.create_right_layout()
        bottom_bar = self.create_bottom_bar()
        columns_layout.addLayout(left_column)
        columns_layout.addLayout(right_column)
        # build the objects together to make the window:
        main_layout.addLayout(columns_layout)
        main_layout.addStretch()
        main_layout.addWidget(bottom_bar)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        menubar = self.menuBar()
        # file menu:
        file_menu = menubar.addMenu('File')
            # load cache action:
        load_cache_action = QAction(QIcon(), "Load cached config", self)
        load_cache_action.setShortcut("Ctrl+C")
        load_cache_action.triggered.connect(self.load_cached_config)
        file_menu.addAction(load_cache_action)
            # load any config action:
        load_any_action = QAction(QIcon(), "Load any config", self)
        load_any_action.setShortcut("Ctrl+Shift+C")
        load_any_action.triggered.connect(self.load_any_config)
        file_menu.addAction(load_any_action)
            # save config action:
        save_action = QAction(QIcon(), "Save config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        # run menu:
        run_menu = menubar.addMenu('Run')
            # run action:
        run_action = QAction(QIcon(), "Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run)
        run_menu.addAction(run_action)
        # window menu:
        window_menu = menubar.addMenu('Window')
            # close all display windows action:
        close_all_display_windows_action = QAction(QIcon(), "Close all display windows", self)
        close_all_display_windows_action.setShortcut("Ctrl+Shift+W")
        close_all_display_windows_action.triggered.connect(DisplayWindowManager.close_all)
        window_menu.addAction(close_all_display_windows_action)

    def create_left_layout(self):
        left_layout = QVBoxLayout()
        # creation of the widget one after another:
        self.input_files_section = InputFilesSection(parent=self)
        self.operator_section = OperatorParametersSection(parent=self)
        self.add_noise_section = AddNoiseSection(parent=self)
        # build the widgets together to make the left layout:
        left_layout.addWidget(self.input_files_section)
        left_layout.addWidget(self.operator_section)
        left_layout.addWidget(self.add_noise_section)
        left_layout.addStretch()
        return left_layout

    def create_right_layout(self):
        right_layout = QVBoxLayout()
        self.algorithm_section = AlgorithmAndAlgoParamsSection(parent=self)
        right_layout.addWidget(self.algorithm_section)
        right_layout.addStretch()
        return right_layout

    def create_bottom_bar(self):
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        self.load_cached_config_button = QPushButton("Load cached config")
        self.load_any_config_button = QPushButton("Load any config")
        self.save_config_button = QPushButton("Save config")
        self.open_recons_button = QPushButton("Open reconstruction")
        self.run_button = QPushButton("Run")
        self.load_cached_config_button.clicked.connect(self.load_cached_config)
        self.load_any_config_button.clicked.connect(self.load_any_config)
        self.open_recons_button.clicked.connect(self.open_reconstruction)
        self.save_config_button.clicked.connect(self.save_config)
        self.run_button.clicked.connect(self.run)
        bottom_layout.addWidget(self.load_cached_config_button)
        bottom_layout.addWidget(self.load_any_config_button)
        bottom_layout.addWidget(self.save_config_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.open_recons_button)
        bottom_layout.addWidget(self.run_button)
        bottom_widget.setLayout(bottom_layout)
        return bottom_widget

    def load_cached_config(self):
        self.input_files_section.update_ui_from_toml(CONFIG_PATH)
        self.operator_section.update_ui_from_toml(CONFIG_PATH)
        self.algorithm_section.update_ui_from_toml(CONFIG_PATH)
        self.add_noise_section.update_ui_from_toml(CONFIG_PATH)

    def load_any_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select a complete config file",
                                                   str(RESULTS_DIR),
                                                   "Configuration Files (*.toml)")
        if file_path:
            # writes the entire config file in the cache:
            config = load_or_create_toml(file_path)
            print(config['algo-params'])
            save_toml(config, CONFIG_PATH)
            # then loads the cache:
            self.load_cached_config()

    def save_config(self):
        save_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Save Parameters File",
                                                   str(RESULTS_DIR),
                                                   "Configuration Files (*.toml)")
        if save_path:
            config = load_or_create_toml(CONFIG_PATH)
            save_toml(config, save_path)

    def open_reconstruction(self):
        open_dir = QFileDialog.getExistingDirectory(self, "Select Result Folder", str(RESULTS_DIR))
        if open_dir not in ['', RESULTS_DIR, None]:
            config = load_or_create_toml(join(open_dir, 'config.toml'))
            f = load_tif(join(open_dir, 'f.TIF'))
            display_window = DisplayWindow(config, f=f)
            messages_section_content = load_txt(join(open_dir, 'messages.txt'))
            display_window.messages_section._print(messages_section_content)
            DisplayWindowManager.add(display_window)
            if config['input-paths']['mode'] == 'synthetic-data':
                f_true = load_tif(join(open_dir, 'f_true.TIF'))
                display_window.f_true = f_true
            display_window.show()
            display_window.update_plot()

    def run(self):
        config = load_or_create_toml(CONFIG_PATH)
        print("Initializing Run with the current configuration.")
        if self.check_missing_parameters_in_config(config):
            # Créer une nouvelle fenêtre
            display_window = DisplayWindow(config)
            DisplayWindowManager.add(display_window)
            display_window.show()
            display_window.run()
        else: print("Running failed.\n")

    @staticmethod
    def check_missing_parameters_in_config(config):
        is_any_parameter_missing = False
        if config['algorithm'] == 'None':
            print("Missing parameter to Run: no 'algorithm' selected.")
            is_any_parameter_missing = True
        if config['input-paths']['tif'] == 'None':
            print("Missing parameter to Run: no '.tif file' selected in 'input-paths' section.")
            is_any_parameter_missing = True
        if config['input-paths']['json'] == 'None':
            print("Missing parameter to Run: no '.json file' selected in 'input-paths' section.")
            is_any_parameter_missing = True
        if config['oper-params']['nz'] == 'None':
            print("Missing parameter to Run: no 'nz' selected in 'oper-params' section.")
            is_any_parameter_missing = True
        if config['oper-params']['z0'] == 'None':
            print("Missing parameter to Run: no 'z0' selected in 'oper-params' section.")
            is_any_parameter_missing = True
        if config['oper-params']['zN'] == 'None':
            print("Missing parameter to Run: no 'zN' selected in 'oper-params' section.")
            is_any_parameter_missing = True
        if config['add-noise']['add_noise']:
            if config['add-noise']['sigma'] == 'None':
                print("Missing parameter to Run: no 'sigma' selected in 'add-noise' section.")
                is_any_parameter_missing = True
        return not is_any_parameter_missing  # if not is_any_parameter_missing == True, then you can run

    def closeEvent(self, event):
        # Fermer d'abord toutes les DisplayWindow
        DisplayWindowManager.close_all()
        if self.input_files_section.json_selector.measurement_parameters_editor is not None:
            # close matirf_parameters_editor window if is it's currently opened
            self.input_files_section.json_selector.measurement_parameters_editor.close()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # escape is clicked
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        elif event.key() == Qt.Key_W and event.modifiers() & Qt.ControlModifier:  # Ctrl+W is clicked
            self.close()
        else:
            super().keyPressEvent(event)
