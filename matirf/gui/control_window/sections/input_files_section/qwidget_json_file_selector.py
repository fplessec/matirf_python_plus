import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from common.gui.file_dialog import open_file

from .measurement_parameters_editor import MeasurementParametersEditor, MEASUREMENT_PARAMETERS_UI
from common.gui.widgets import QCrossButton
from matirf import MATIRF_MEASUREMENTS_DIR
from common.in_out import load_json
from matirf.cache import update_cache
from common.settings import FontSize


def check_measurement_parameters_file_format(filepath):
    """This function ensures that the selected JSON file is written correctly."""
    dictionary = load_json(filepath)
    REQUIRED_KEYS = list(MEASUREMENT_PARAMETERS_UI.keys())
    missing_keys = []
    for key in REQUIRED_KEYS:
        try:
            _ = dictionary[key]
        except KeyError:
            missing_keys.append(key)
    if missing_keys:  # if this list is not empty ie if there are missing keys
        print(f"Failed to select the following file: '{filepath}'\n"
              f"-> Invalid measurement parameters file format. This file is missing required key(s): "
              f"{', '.join(missing_keys)}")
    return not missing_keys


class JsonFileSelector(QWidget):
    """
    A widget that allows the user to select a measurement parameters file with a button. The labelling of the widget
    changes with its parent attribute 'mode1', and its parent is InputFilesSection. A second button allows the user to
    open a MeasurementParametersEditor to edit or create a measurement parameters file.

    This QWidget focus on registering the desired JSON file path (measurement parameters of the data) inside the
    [input-paths][json] key of the cached config.toml file.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.is_file_selected = False
        self.measurement_parameters_editor = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(10)
        # creation of the widgets one after another:
        self.title_label = QLabel(self.mode_dependent_text_update())
        self.title_label.setStyleSheet(f"font-size: {FontSize.NORMAL}pt;")
        choose_button = QPushButton("Choose .json file")
        choose_button.clicked.connect(self.choose_file)
        self.create_modify_button = QPushButton()
        self.update_create_modify_button_text()
        self.create_modify_button.clicked.connect(self.create_modify_file)
        self.file_label = QLabel("No .json file selected")
        file_label_color = self.palette().color(QPalette.PlaceholderText).name()
        self.file_label.setStyleSheet(f"color: {file_label_color}; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        unselect_button = QCrossButton()
        unselect_button.clicked.connect(self.unselect_file)
        last_line = QHBoxLayout()
        # build the widgets together to make the layout:
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_button, alignment=Qt.AlignHCenter)
        layout.addWidget(self.create_modify_button, alignment=Qt.AlignHCenter)
        layout.addStretch()
        last_line.addStretch()
        last_line.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        last_line.addWidget(unselect_button)
        last_line.addStretch()
        layout.addLayout(last_line)
        self.setLayout(layout)

    def choose_file(self):
        file_path = open_file(self, self.mode_dependent_text_update(),
                              MATIRF_MEASUREMENTS_DIR, "Parameters Files (*.json)")
        if file_path:
            if check_measurement_parameters_file_format(file_path):
                self.update_selected_file(file_path)

    def update_selected_file(self, file_path):
        self.json_path = file_path
        update_cache(["input-paths", "json"], file_path)  # -> to cache
        file_name = os.path.basename(file_path)
        self.is_file_selected = file_path != 'None'
        self.update_ui(file_name)

    def unselect_file(self):
        update_cache(["input-paths", "json"], 'None')  # -> to cache
        self.is_file_selected = False
        self.update_ui('None')

    def update_ui(self, file_name):
        self.file_label.setText(f"selected file : {file_name}" if self.is_file_selected else "No .json file selected")
        font = self.file_label.font()
        font.setBold(self.is_file_selected)
        self.file_label.setFont(font)
        self.update_create_modify_button_text()
        self.parent.image_selector.preprocess_button.setVisible(self.parent.are_both_file_selected())

    def update_create_modify_button_text(self):
        self.create_modify_button.setText("Modify .json file" if self.is_file_selected else "Create .json file")

    def create_modify_file(self):
        if self.measurement_parameters_editor is not None:
            self.measurement_parameters_editor.close()
        self.measurement_parameters_editor = MeasurementParametersEditor(parent=self)
        self.measurement_parameters_editor.show()

    def mode_dependent_text_update(self):
        return "Path of the measurement parameters" if self.parent.is_mode_real\
               else "Path of the simulated parameters"

    def update_mode(self):
        self.title_label.setText(self.mode_dependent_text_update())
