import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QMessageBox,
)

from gui.more_widgets import QSwitchButton, QSeparator, QCrossButton
from base import DataMode
from deconv.in_out import load_or_create_toml, DECONV_CONFIG_PATH, DECONV_MEASUREMENTS_DIR
from deconv.cache import update_cache
from settings import FontSize


## widget to select a PNG file (2D image for deconv):
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
        path, _ = QFileDialog.getOpenFileName(
            self, self._mode_text(), str(DECONV_MEASUREMENTS_DIR), "Image Files (*.png)")
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


## widget to select a JSON file (PSF / simulation parameters for deconv):
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
        path, _ = QFileDialog.getOpenFileName(
            self, self._mode_text(), str(DECONV_MEASUREMENTS_DIR), "Parameters Files (*.json)")
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
        self.parent.png_selector.preprocess_button.setVisible(self.parent.are_both_file_selected())

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


## input files section for the deconv problem (PNG + JSON with real/synthetic switch):
class DeconvInputFilesSection(QGroupBox):

    def __init__(self, parent=None):
        super().__init__("Input Files")
        self.on_color = self.palette().color(QPalette.WindowText).name()
        self.off_color = self.palette().color(QPalette.PlaceholderText).name()
        self.parent = parent
        self.is_mode_real = self._get_cached_mode()
        self._setup_ui()

    @staticmethod
    def _get_cached_mode():
        try:
            mode = load_or_create_toml(DECONV_CONFIG_PATH)['input-paths']['mode']
            return mode == DataMode.REAL.value
        except (KeyError, FileNotFoundError):
            return False  # default is synthetic for deconv

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addLayout(self._create_top_layout())
        layout.addLayout(self._create_bottom_layout())
        self.setLayout(layout)

    def _create_top_layout(self):
        top = QHBoxLayout()
        top.setContentsMargins(4, 4, 4, 4)
        self.switch_button = QSwitchButton()
        if not self.is_mode_real:
            self.switch_button.switch_to_right()
        self.switch_button.toggled.connect(self._switch_mode)
        self.real_label = QLabel("Work with real measurement")
        self.synth_label = QLabel("Simulate with synthetic truth")
        self._update_labels()
        top.addStretch()
        top.addWidget(self.real_label)
        top.addWidget(self.switch_button)
        top.addWidget(self.synth_label)
        top.addStretch()
        return top

    def _create_bottom_layout(self):
        bot = QHBoxLayout()
        bot.setContentsMargins(1, 1, 1, 1)
        self.png_selector = PngFileSelector(parent=self)
        self.json_selector = DeconvJsonFileSelector(parent=self)
        bot.addWidget(self.png_selector)
        bot.addWidget(QSeparator('V'))
        bot.addWidget(self.json_selector)
        return bot

    def are_both_file_selected(self):
        return self.png_selector.is_file_selected and self.json_selector.is_file_selected

    def _update_labels(self):
        if self.is_mode_real:
            self.real_label.setStyleSheet(
                f"color: {self.on_color}; font-style: italic; font-size: {FontSize.SMALL}pt;")
            self.synth_label.setStyleSheet(
                f"color: {self.off_color}; font-style: italic; font-size: {FontSize.SMALL}pt;")
        else:
            self.real_label.setStyleSheet(
                f"color: {self.off_color}; font-style: italic; font-size: {FontSize.SMALL}pt;")
            self.synth_label.setStyleSheet(
                f"color: {self.on_color}; font-style: italic; font-size: {FontSize.SMALL}pt;")

    def _switch_mode(self):
        self.is_mode_real = not self.is_mode_real
        self._update_labels()
        new_mode = DataMode.REAL.value if self.is_mode_real else DataMode.SYNTHETIC.value
        update_cache(['input-paths', 'mode'], new_mode)
        self.png_selector.update_mode()
        self.json_selector.update_mode()

    def update_ui_from_toml(self, toml_path):
        config = load_or_create_toml(toml_path)
        mode = config['input-paths']['mode']
        self.is_mode_real = mode == DataMode.REAL.value
        self._update_labels()
        self.png_selector.update_mode()
        self.json_selector.update_mode()
        if self.is_mode_real:
            self.switch_button.switch_to_left(no_signal=True)
        else:
            self.switch_button.switch_to_right(no_signal=True)
        self.png_selector.update_selected_file(config['input-paths']['png'])
        self.json_selector.update_selected_file(config['input-paths']['json'])
