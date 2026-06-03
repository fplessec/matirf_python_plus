from os.path import join

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QAction

from .sections import DeconvInputFilesSection, DeconvAlgorithmSection, DeconvAddNoiseSection
from .display_window_manager import DeconvDisplayWindowManager
from gui.utils import close_active_window
from deconv.in_out import load_or_create_toml, DECONV_CONFIG_PATH, DECONV_RESULTS_DIR, save_toml
from deconv.core import DeconvPipelineManager
import settings


## main window for the deconv problem (no operator parameters, PNG input):
class DeconvControlWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deconvolution Parameter Selection")
        self.setGeometry(100, 100, settings.width_cw, settings.height_cw)
        self._setup_ui()
        self.load_cached_config()

    def _setup_ui(self):
        self.setFocus()
        self._create_menu_bar()
        central = QWidget()
        main_layout = QVBoxLayout()
        columns = QHBoxLayout()
        columns.addLayout(self._create_left_layout())
        columns.addLayout(self._create_right_layout())
        main_layout.addLayout(columns)
        main_layout.addStretch()
        main_layout.addWidget(self._create_bottom_bar())
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        load_any = QAction(QIcon(), "Load any config", self)
        load_any.setShortcut("Ctrl+C")
        load_any.triggered.connect(self._load_any_config)
        file_menu.addAction(load_any)
        save_action = QAction(QIcon(), "Save config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)
        recon_menu = menubar.addMenu('Reconstruction')
        run_action = QAction(QIcon(), "Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self._run)
        recon_menu.addAction(run_action)
        open_action = QAction(QIcon(), "Open reconstruction", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_reconstruction)
        recon_menu.addAction(open_action)
        window_menu = menubar.addMenu('Window')
        close_this = QAction(QIcon(), "Close this window", self)
        close_this.setShortcut("Ctrl+W")
        close_this.triggered.connect(close_active_window)
        window_menu.addAction(close_this)
        close_all = QAction(QIcon(), "Close all reconstruction windows", self)
        close_all.setShortcut("Ctrl+Shift+W")
        close_all.triggered.connect(DeconvDisplayWindowManager.close_all)
        window_menu.addAction(close_all)

    def _create_left_layout(self):
        layout = QVBoxLayout()
        self.input_files_section = DeconvInputFilesSection(self)
        self.add_noise_section = DeconvAddNoiseSection(self)
        layout.addWidget(self.input_files_section)
        layout.addWidget(self.add_noise_section)
        layout.addStretch()
        return layout

    def _create_right_layout(self):
        layout = QVBoxLayout()
        self.algorithm_section = DeconvAlgorithmSection(self)
        layout.addWidget(self.algorithm_section)
        layout.addStretch()
        return layout

    def _create_bottom_bar(self):
        widget = QWidget()
        layout = QHBoxLayout()
        btn_load = QPushButton("Load any config")
        btn_load.clicked.connect(self._load_any_config)
        btn_save = QPushButton("Save config")
        btn_save.clicked.connect(self._save_config)
        btn_open = QPushButton("Open reconstruction")
        btn_open.clicked.connect(self._open_reconstruction)
        btn_run = QPushButton("Run")
        btn_run.clicked.connect(self._run)
        layout.addWidget(btn_load)
        layout.addWidget(btn_save)
        layout.addStretch()
        layout.addWidget(btn_open)
        layout.addWidget(btn_run)
        widget.setLayout(layout)
        return widget

    ## loads each section's UI from the cached TOML config:
    def load_cached_config(self):
        self.input_files_section.update_ui_from_toml(DECONV_CONFIG_PATH)
        self.algorithm_section.update_ui_from_toml(DECONV_CONFIG_PATH)
        self.add_noise_section.update_ui_from_toml(DECONV_CONFIG_PATH)

    def _load_any_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select config", str(DECONV_RESULTS_DIR), "*.toml")
        if path:
            config = load_or_create_toml(path)
            save_toml(config, DECONV_CONFIG_PATH)
            self.load_cached_config()

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", str(DECONV_RESULTS_DIR), "*.toml")
        if path:
            config = load_or_create_toml(DECONV_CONFIG_PATH)
            save_toml(config, path)

    ## creates and starts a deconv pipeline from the cached config, then opens a display window:
    def _run(self):
        config = load_or_create_toml(DECONV_CONFIG_PATH)
        pipeline = DeconvPipelineManager.create(config)
        display_window = DeconvDisplayWindowManager.create(pipeline)
        display_window.show()
        pipeline.start()

    ## loads a saved reconstruction from a folder and displays it:
    def _open_reconstruction(self):
        open_dir = QFileDialog.getExistingDirectory(
            self, "Select Folder", str(DECONV_RESULTS_DIR))
        if not open_dir:
            return
        config = load_or_create_toml(join(open_dir, 'config.toml'))
        pipeline = DeconvPipelineManager.create(config)
        pipeline.load_results(open_dir)
        display_window = DeconvDisplayWindowManager.create(pipeline)
        display_window.show()
        display_window.initialize_from_existing_data()

    ## closes all sub-windows and stops all pipelines before quitting:
    def closeEvent(self, event):
        if self.input_files_section.json_selector.psf_parameters_editor:
            self.input_files_section.json_selector.psf_parameters_editor.close()
        if self.input_files_section.png_selector.preprocess_viewer:
            self.input_files_section.png_selector.preprocess_viewer.close()
        DeconvDisplayWindowManager.close_all()
        DeconvPipelineManager.stop_all()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            widget = self.focusWidget()
            if widget:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)
