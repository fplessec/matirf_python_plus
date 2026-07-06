import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QMessageBox,
)
from common.gui.file_dialog import open_file

from common.gui.widgets import QCrossButton
from common.gui.specializable.control_window import BaseInputFilesSection
from common import DataMode
from deconv import DECONV_CONFIG_PATH, DECONV_MEASUREMENTS_DIR, DEFAULT_DECONV_CONFIG
from common.in_out import load_or_create_toml
from deconv.cache import update_cache
from common.settings import FontSize


# ── widget to select a PNG file (2D image for deconv) ───────────────────

class PngFileSelector(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.is_file_selected = False
        self.preprocess_viewer = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(10)
        self.title_label = QLabel(self._mode_text())
        self.title_label.setStyleSheet(f"font-size: {FontSize.NORMAL}pt;")
        choose_btn = QPushButton("Choose .png file")
        choose_btn.clicked.connect(self._choose_file)
        self.preprocess_button = QPushButton("See preprocessed file")
        self.preprocess_button.clicked.connect(self._open_preprocess_viewer)
        self.preprocess_button.setVisible(False)
        self.file_label = QLabel("No .png file selected")
        placeholder_color = self.palette().color(QPalette.PlaceholderText).name()
        self.file_label.setStyleSheet(
            f"color: {placeholder_color}; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        unselect_btn = QCrossButton()
        unselect_btn.clicked.connect(self._unselect)
        last_line = QHBoxLayout()
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.preprocess_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        last_line.addStretch()
        last_line.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        last_line.addWidget(unselect_btn)
        last_line.addStretch()
        layout.addLayout(last_line)
        self.setLayout(layout)

    def _choose_file(self):
        path = open_file(self, self._mode_text(), DECONV_MEASUREMENTS_DIR, "Image Files (*.png)")
        if path:
            self.update_selected_file(path)

    def update_selected_file(self, path):
        update_cache(["input-paths", "png"], path)
        self.is_file_selected = path != 'None'
        self._update_ui(os.path.basename(path))

    def _unselect(self):
        update_cache(["input-paths", "png"], 'None')
        self.is_file_selected = False
        self._update_ui('None')

    def _update_ui(self, file_name):
        self.file_label.setText(
            f"selected file : {file_name}" if self.is_file_selected else "No .png file selected")
        font = self.file_label.font()
        font.setBold(self.is_file_selected)
        self.file_label.setFont(font)
        self.preprocess_button.setVisible(self.parent.are_both_file_selected())

    def _open_preprocess_viewer(self):
        from .qwidget_preprocess_viewer import DeconvPreprocessViewer
        if self.preprocess_viewer is not None:
            self.preprocess_viewer.close()
        try:
            self.preprocess_viewer = DeconvPreprocessViewer(parent=self)
        except Exception as e:
            config = load_or_create_toml(DECONV_CONFIG_PATH, default_config=DEFAULT_DECONV_CONFIG)
            errors = []
            if config['add-noise']['add_noise'] and config['add-noise']['sigma'] == 'None':
                errors.append('Noise standard deviation value is None, please define a value for this parameter.')
            if errors:
                e = Exception('\n-'+'\n-'.join(errors))
            self.preprocess_viewer = None
            QMessageBox.warning(
                self,
                "Cannot preview the preprocessing",
                f"An unexpected error occurred while computing the preview:\n\n"
                f"{type(e).__name__}: {e}")
            return
        self.preprocess_viewer.show()

    def _mode_text(self):
        return ("Path of the 2D image (measurement)"
                if self.parent.is_mode_real
                else "Path of the 2D ground truth")

    def update_mode(self):
        self.title_label.setText(self._mode_text())


# ── widget to select a JSON file (PSF / simulation parameters) ──────────

class DeconvJsonFileSelector(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.is_file_selected = False
        self.json_path = None
        self.psf_parameters_editor = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(10)
        self.title_label = QLabel(self._mode_text())
        self.title_label.setStyleSheet(f"font-size: {FontSize.NORMAL}pt;")
        choose_btn = QPushButton("Choose .json file")
        choose_btn.clicked.connect(self._choose_file)
        self.create_modify_button = QPushButton()
        self._update_create_modify_button_text()
        self.create_modify_button.clicked.connect(self._create_modify_file)
        self.file_label = QLabel("No .json file selected")
        placeholder_color = self.palette().color(QPalette.PlaceholderText).name()
        self.file_label.setStyleSheet(
            f"color: {placeholder_color}; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        unselect_btn = QCrossButton()
        unselect_btn.clicked.connect(self._unselect)
        last_line = QHBoxLayout()
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.create_modify_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        last_line.addStretch()
        last_line.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        last_line.addWidget(unselect_btn)
        last_line.addStretch()
        layout.addLayout(last_line)
        self.setLayout(layout)

    def _choose_file(self):
        path = open_file(self, self._mode_text(), DECONV_MEASUREMENTS_DIR, "Parameters Files (*.json)")
        if path:
            self.update_selected_file(path)

    def update_selected_file(self, path):
        self.json_path = path
        update_cache(["input-paths", "json"], path)
        self.is_file_selected = path != 'None'
        self._update_ui(os.path.basename(path))

    def _unselect(self):
        update_cache(["input-paths", "json"], 'None')
        self.json_path = None
        self.is_file_selected = False
        self._update_ui('None')

    def _update_ui(self, file_name):
        self.file_label.setText(
            f"selected file : {file_name}" if self.is_file_selected else "No .json file selected")
        font = self.file_label.font()
        font.setBold(self.is_file_selected)
        self.file_label.setFont(font)
        self._update_create_modify_button_text()
        self.parent.image_selector.preprocess_button.setVisible(self.parent.are_both_file_selected())

    def _update_create_modify_button_text(self):
        self.create_modify_button.setText(
            "Modify .json file" if self.is_file_selected else "Create .json file")

    def _create_modify_file(self):
        from .psf_parameters_editor import PsfParametersEditor
        if self.psf_parameters_editor is not None:
            self.psf_parameters_editor.close()
        self.psf_parameters_editor = PsfParametersEditor(parent=self)
        self.psf_parameters_editor.show()

    def _mode_text(self):
        return ("Path of the PSF parameters"
                if self.parent.is_mode_real
                else "Path of the simulation parameters")

    def update_mode(self):
        self.title_label.setText(self._mode_text())


# ── input files section for the deconv problem ──────────────────────────

class DeconvInputFilesSection(BaseInputFilesSection):
    """
    Deconvolution input files section.

    Selects a .png file (2D image or ground truth) and a .json PSF
    parameters file, with a real / synthetic mode toggle.
    """

    def _get_cached_mode(self) -> bool:
        try:
            mode = load_or_create_toml(
                DECONV_CONFIG_PATH, default_config=DEFAULT_DECONV_CONFIG
            )['input-paths']['mode']
            return mode == DataMode.REAL.value
        except (KeyError, FileNotFoundError):
            return False  # default is synthetic for deconv

    def _real_mode_text(self):
        return "Work with real measurement"

    def _synthetic_mode_text(self):
        return "Simulate with synthetic truth"

    def _create_image_selector(self):
        return PngFileSelector(parent=self)

    def _create_json_selector(self):
        return DeconvJsonFileSelector(parent=self)

    def _on_switch_mode(self, is_mode_real):
        new_mode = DataMode.REAL.value if is_mode_real else DataMode.SYNTHETIC.value
        update_cache(['input-paths', 'mode'], new_mode)

    def _load_config_for_update(self, toml_path):
        return load_or_create_toml(toml_path, default_config=DEFAULT_DECONV_CONFIG)

    def _get_file_paths_from_config(self, config):
        return config['input-paths']['png'], config['input-paths']['json']
