import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout

from gui.more_widgets import QCrossButton
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
        self.is_file_selected = False
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

        self.info_button = QPushButton("Info ...")
        #self.info_button.clicked.connect(self...........)
        self.info_button.setVisible(False)  # hidden by default

        self.file_label = QLabel("No .tif file selected")
        self.file_label.setStyleSheet(f"color: gray; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        unselect_button = QCrossButton()
        unselect_button.clicked.connect(self.unselect_file)
        last_line = QHBoxLayout()
        # build the widgets together to make the layout:
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_button, alignment=Qt.AlignHCenter)
        layout.addWidget(self.info_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        last_line.addStretch()
        last_line.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        last_line.addWidget(unselect_button)
        last_line.addStretch()
        layout.addLayout(last_line)
        self.setLayout(layout)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   self.mode_dependent_text_update(),
                                                   str(MEASUREMENTS_DIR),
                                                   "Image Files (*.tif *.tiff)")
        if file_path:
            self.update_selected_file(file_path)
            # ICI TEST POUR OUVRIR AVEC FIJI:
            # # a faire en bcp mieux etc etc:::
            # import subprocess
            # from settings import fiji_path
            # print(fiji_path)
            # print(fiji_path)
            # subprocess.Popen([str(fiji_path), str(file_path)])

    def update_selected_file(self, file_path):
        self.tif_path = file_path
        update_cache(["input-paths", "tif"], file_path)
        filename = os.path.basename(file_path)
        if file_path != 'None':
            self.is_file_selected = True
            self.file_label.setText(f"selected file : {filename}")
            self.info_button.setVisible(True)
        else:
            self.is_file_selected = False
            self.file_label.setText("No .json file selected")
            self.info_button.setVisible(False)

    def unselect_file(self):
        self.is_file_selected = False
        self.file_label.setText("No .json file selected")
        self.info_button.setVisible(False)
        update_cache(["input-paths", "tif"], 'None')

    def mode_dependent_text_update(self):
        return "Path of the MA-TIRF image stack" if self.parent.mode1 else "Path of the 3D object (synthetic truth)"

    def update_mode(self):
        self.title_label.setText(self.mode_dependent_text_update())