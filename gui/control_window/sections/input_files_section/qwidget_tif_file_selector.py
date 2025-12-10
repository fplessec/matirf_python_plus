import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog

from settings import FontSize
from cache import update_cache
from in_out import MEASUREMENTS_DIR


class TifFileSelector(QWidget):
    """
    A widget that allows the user to select a tif file with a button. The labelling of the widget changes with its
    parent attribute 'mode1', and its parent is InputFilesSection. In mode1, the user have to select a real MA-TIRF
    measurement. In mode2 (no mode1), the user have to select a 3D image that will be used as a ground truth in order
    to create a synthetic MA-TIRF measurement.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setup_ui()
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        # creation of the widgets one after another:
        self.title_label = QLabel(self.mode_dependent_text_update())
        self.title_label.setStyleSheet(f"font-size: {FontSize.NORMAL}pt;")
        choose_button = QPushButton("Choose .tif file")
        choose_button.clicked.connect(self.choose_file)
        self.file_label = QLabel("No .tif file selected")
        self.file_label.setStyleSheet(f"color: gray; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        # build the widgets together to make the layout:
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        self.setLayout(layout)
    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   self.mode_dependent_text_update(),
                                                   str(MEASUREMENTS_DIR),
                                                   "Image Files (*.tif *.tiff)")
        if file_path:
            self.update_selected_file(file_path)
    def update_selected_file(self, file_path):
        self.tif_path = file_path
        update_cache(["input-paths", "tif"], file_path)
        filename = os.path.basename(file_path)
        self.file_label.setText(f"selected file : {filename}")
    def mode_dependent_text_update(self):
        return "Path of the MA-TIRF image stack" if self.parent.mode1 else "Path of the 3D object (synthetic truth)"
    def update_mode(self):
        self.title_label.setText(self.mode_dependent_text_update())