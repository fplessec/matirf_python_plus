import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout

from .qwidget_tif_file_preprocess_viewer import TifFilePreprocessViewer
from gui.more_widgets import QCrossButton
from in_out import MEASUREMENTS_DIR
from cache import update_cache
from settings import FontSize


class TifFileSelector(QWidget):
    """
    A widget that allows the user to select a tif file with a button. The labelling of the widget changes with its
    parent attribute 'mode1', and its parent is InputFilesSection. In mode1, the user have to select a real MA-TIRF
    measurement. In mode2 (no mode1), the user have to select a 3D image that will be used as a ground truth in order
    to create a synthetic MA-TIRF measurement.

    This QWidget focus on registering the desired TIF file path (MA-TIRF multi-stack data or synthetic 3D truth) inside
    the [input-paths][tif] key of the cached config.toml file.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.is_file_selected = False
        self.tif_file_preprocess_editor = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(10)
        # creation of the widgets one after another:
        self.title_label = QLabel(self.mode_dependent_text_update())
        self.title_label.setStyleSheet(f"font-size: {FontSize.NORMAL}pt;")
        choose_button = QPushButton("Choose .tif file")
        choose_button.clicked.connect(self.choose_file)
        self.preprocess_button = QPushButton("See preprocessed file")
        self.preprocess_button.clicked.connect(self.open_tif_file_preprocess_editor)
        self.preprocess_button.setVisible(False)  # hidden by default
        self.file_label = QLabel("No .tif file selected")
        file_label_color = self.palette().color(QPalette.PlaceholderText).name()
        self.file_label.setStyleSheet(f"color: {file_label_color}; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        unselect_button = QCrossButton()
        unselect_button.clicked.connect(self.unselect_file)
        last_line = QHBoxLayout()
        # build the widgets together to make the layout:
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_button, alignment=Qt.AlignHCenter)
        layout.addWidget(self.preprocess_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        last_line.addStretch()
        last_line.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        last_line.addWidget(unselect_button)
        last_line.addStretch()
        layout.addLayout(last_line)
        self.setLayout(layout)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.mode_dependent_text_update(),
                                                   str(MEASUREMENTS_DIR), "Image Files (*.tif *.tiff)")
        if file_path:
            self.update_selected_file(file_path)

    def update_selected_file(self, file_path):
        self.tif_path = file_path
        update_cache(["input-paths", "tif"], file_path)  # -> to cache
        file_name = os.path.basename(file_path)
        self.is_file_selected = file_path != 'None'
        self.update_ui(file_name)

    def unselect_file(self):
        update_cache(["input-paths", "tif"], 'None')  # -> to cache
        self.is_file_selected = False
        self.update_ui('None')

    def update_ui(self, file_name):
        self.file_label.setText(f"selected file : {file_name}" if self.is_file_selected else "No .tif file selected")
        font = self.file_label.font()
        font.setBold(self.is_file_selected)
        self.file_label.setFont(font)
        self.preprocess_button.setVisible(self.parent.are_both_file_selected())  # method from InputFilesSection

    def open_tif_file_preprocess_editor(self):
        if self.tif_file_preprocess_editor is not None:
            self.tif_file_preprocess_editor.close()
        self.tif_file_preprocess_editor = TifFilePreprocessViewer(parent=self)
        self.tif_file_preprocess_editor.show()

    def mode_dependent_text_update(self):
        return "Path of the MA-TIRF image stack" if self.parent.mode1 else "Path of the 3D object (synthetic truth)"

    def update_mode(self):
        self.title_label.setText(self.mode_dependent_text_update())