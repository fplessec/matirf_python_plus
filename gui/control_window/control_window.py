from os.path import join

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QAction

from .sections import InputFilesSection, OperatorParametersSection, AlgorithmAndAlgoParamsSection, AddNoiseSection
from .display_window_manager import DisplayWindowManager
from gui.utils import close_active_window
from in_out import load_or_create_toml, CONFIG_PATH, RESULTS_DIR, save_toml
from core import PipelineManager
import settings


class ControlWindow(QMainWindow):
    """
    Main window of the U.I., allows the user to choose all parameters before running a reconstruction.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MA-TIRF Parameter Selection")
        self.setGeometry(100, 100, settings.width_cw, settings.height_cw)
        self.setup_ui()
        # if it's the first time the config file it's create, allow it no fill the necessary parameters to 'None',
        # and to put the bool parameters to their default value:
        self.load_cached_config()

    ## builds the main layout with menu bar, two columns, and bottom bar:
    def setup_ui(self):
        self.setFocus()
        self.create_menu_bar()
        # the base objects:
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        columns = QHBoxLayout()
        # create and assemble the specifics layouts one after another:
        columns.addLayout(self.create_left_layout())
        columns.addLayout(self.create_right_layout())
        # build the objects together to make the window:
        main_layout.addLayout(columns)
        main_layout.addStretch()
        main_layout.addWidget(self.create_bottom_bar())
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        menubar = self.menuBar()
        # file menu:
        file_menu = menubar.addMenu('File')
            # load config action:
        load_any = QAction(QIcon(), "Load any config", self)
        load_any.setShortcut("Ctrl+C")
        load_any.triggered.connect(self.load_any_config)
        file_menu.addAction(load_any)
            # save config action:
        save_action = QAction(QIcon(), "Save config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        # reconstruction menu:
        reconstruction_menu = menubar.addMenu('Reconstruction')
            # run action:
        run_action = QAction(QIcon(), "Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run)
        reconstruction_menu.addAction(run_action)
            # open reconstruction action:
        open_recons_action = QAction(QIcon(), "Open reconstruction", self)
        open_recons_action.setShortcut("Ctrl+O")
        open_recons_action.triggered.connect(self.open_reconstruction)
        reconstruction_menu.addAction(open_recons_action)
        # window menu:
        window_menu = menubar.addMenu('Window')
            # close active window action:
        close_this = QAction(QIcon(), "Close this window", self)
        close_this.setShortcut("Ctrl+W")
        close_this.triggered.connect(close_active_window)
        window_menu.addAction(close_this)
            # close all display windows action:
        close_all = QAction(QIcon(), "Close all reconstruction windows", self)
        close_all.setShortcut("Ctrl+Shift+W")
        close_all.triggered.connect(DisplayWindowManager.close_all)
        window_menu.addAction(close_all)

    def create_left_layout(self):
        layout = QVBoxLayout()
        # creation of the widget one after another:
        self.input_files_section = InputFilesSection(self)
        self.operator_section = OperatorParametersSection(self)
        self.add_noise_section = AddNoiseSection(self)
        # build the widgets together to make the left layout:
        layout.addWidget(self.input_files_section)
        layout.addWidget(self.operator_section)
        layout.addWidget(self.add_noise_section)
        layout.addStretch()
        return layout

    def create_right_layout(self):
        layout = QVBoxLayout()
        # creation of the widget one after another:
        self.algorithm_section = AlgorithmAndAlgoParamsSection(self)
        # build the widgets together to make the right layout:
        layout.addWidget(self.algorithm_section)
        layout.addStretch()
        return layout

    def create_bottom_bar(self):
        widget = QWidget()
        layout = QHBoxLayout()
        # creation of the widget one after another:
        btn_load_any = QPushButton("Load any config")
        btn_load_any.clicked.connect(self.load_any_config)
        btn_save = QPushButton("Save config")
        btn_save.clicked.connect(self.save_config)
        btn_open = QPushButton("Open reconstruction")
        btn_open.clicked.connect(self.open_reconstruction)
        btn_run = QPushButton("Run")
        btn_run.clicked.connect(self.run)
        # build the widgets together to make the bottom bar widget:
        layout.addWidget(btn_load_any)
        layout.addWidget(btn_save)
        layout.addStretch()
        layout.addWidget(btn_open)
        layout.addWidget(btn_run)
        widget.setLayout(layout)
        return widget

    ## loads each section's UI from the cached TOML config:
    def load_cached_config(self):
        self.input_files_section.update_ui_from_toml(CONFIG_PATH)  # <- from cache
        self.operator_section.update_ui_from_toml(CONFIG_PATH)  # <- from cache
        self.algorithm_section.update_ui_from_toml(CONFIG_PATH)  # <- from cache
        self.add_noise_section.update_ui_from_toml(CONFIG_PATH)  # <- from cache

    def load_any_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select config", str(RESULTS_DIR), "*.toml")
        if file_path:
            # writes the entire config file in the cache:
            config = load_or_create_toml(file_path)  # <- from any config path
            save_toml(config, CONFIG_PATH)  # -> to cache
            # then loads the cache:
            self.load_cached_config()  # <- from cache

    def save_config(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save config", str(RESULTS_DIR), "*.toml")
        if save_path:
            config = load_or_create_toml(CONFIG_PATH)  # <- from cache
            save_toml(config, save_path)  # -> to any config path

    ## creates and starts a pipeline from the cached config, then opens a display window:
    def run(self):
        config = load_or_create_toml(CONFIG_PATH)  # <- from cache
        pipeline = PipelineManager.create(config)
        display_window = DisplayWindowManager.create(pipeline)
        display_window.show()
        pipeline.start()

    ## loads a saved reconstruction from a folder and displays it:
    def open_reconstruction(self):
        open_dir = QFileDialog.getExistingDirectory(self, "Select Folder", str(RESULTS_DIR))
        if not open_dir: return
        config = load_or_create_toml(join(open_dir, 'config.toml'))  # -> to recons folder
        pipeline = PipelineManager.create(config)
        pipeline.load_results(open_dir)
        display_window = DisplayWindowManager.create(pipeline)
        display_window.show()
        display_window.initialize_from_existing_data()

    ## closes all sub-windows and stops all pipelines before quitting:
    def closeEvent(self, event):
        if self.input_files_section.json_selector.measurement_parameters_editor:
            self.input_files_section.json_selector.measurement_parameters_editor.close()
        if self.input_files_section.tif_selector.tif_file_preprocess_editor:
            self.input_files_section.tif_selector.tif_file_preprocess_editor.close()
        DisplayWindowManager.close_all()
        PipelineManager.stop_all()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # escape is pressed
            widget = self.focusWidget()
            if widget:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)