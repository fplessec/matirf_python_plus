from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QFileDialog, QMessageBox

from .measurement_parameters_ui_dictionary import MEASUREMENT_PARAMETERS_UI
from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator, QTextEditTab2Switch
from in_out import load_json, save_json, MEASUREMENTS_DIR
import settings


class MeasurementParametersEditor(QWidget):
    """
    A widget that allows the user to choose the measurement parameters values from an existing measurement parameters
    file, or to same them in a new one. This file is a json file, and should contain those parameters:
        > "angles_deg": a list of the incident angle for each measurement stack, in degrees
        > "n_glass": the optical index of the incident medium
        > "n_medium": the optical index of the sample medium
        > "n_oil": the optical index of the immersion oil of the objective
        > "numerical_aperture": the numerical aperture of the objective
        > "wavelength_nm": the wavelength of the excitation light, in nanometers
        > "beam_divergence_deg": the divergence of the excitation beam, in degrees
    This widget opens in a new window and uses mostly SimpleParameterWidget(s) to manage parameters. We use a dedicated
    dictionary MEASUREMENT_PARAMETERS_UI that stores all the attributes for each SimpleParameterWidget.

    The U.I. consists of :
        > first line, a QLabel to give information about the next widget
        > a QWidget to write the list of incident angle
        > each line then is a SimpleParameterWidget for the other parameters
        > a QPushButton to save the parameters in the json file
    """
    def __init__(self, parent):
        super().__init__()
        self.params_ui_dict = MEASUREMENT_PARAMETERS_UI
        self.parameter_widgets = {}  # <-to keep each widget in memory
        self.parent = parent
        self.json_path = self.parent.json_path if self.parent.is_file_selected else None
        self.json_file = load_json(self.json_path) if self.parent.is_file_selected else None
        self.setWindowTitle("Modify Parameters" if self.parent.is_file_selected else "Create Parameters")
        self.resize(700 , 1)
        self.setup_ui()
        self.update_ui_current_parameters()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 10)
        layout.setSpacing(3)
        # create and put the objects one after another in the layout:
        self.angles_widget = self.create_angle_widget()
        layout.addWidget(self.angles_widget)
        layout.addWidget(QSeparator('H'))
        widgets_count = 0
        for param_name, param_config in self.params_ui_dict.items():
            if widgets_count > 0:
                layout.addWidget(QSeparator('H'))
            if self.parent.is_file_selected:  # if file already selected put values from the file and not default ones
                param_config["param_info"]["default"] = self.json_file[param_name]
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"]
            )
            self.parameter_widgets[param_name] = widget
            if widgets_count == len(self.params_ui_dict) - 1:
                widget.setContentsMargins(1, 1, 1, 3)  # <- add a bottom margin before the button
            layout.addWidget(widget)
            widgets_count += 1
        layout.addWidget(self.create_button())
        self.setLayout(layout)

    def create_angle_widget(self, fontsize=settings.FontSize.NORMAL):
        layout = QVBoxLayout()
        # creation of the objects one after another:
        font = QFont()
        font.setPointSize(fontsize)
        first_line = self.create_first_line(qfont=font)
        input_box_widget = QTextEditTab2Switch(parent=self)
        if self.parent.is_file_selected:
            angles_str = "\n".join([str(angle) for angle in self.json_file['angles_deg']])
            input_box_widget.setPlainText(angles_str)
        input_box_widget.setFont(font)
        # build the objects together to make the layout:
        layout.addLayout(first_line)
        layout.addWidget(input_box_widget)
        # wrap the layout in a QWidget
        container = QWidget()
        container.setLayout(layout)
        return container

    def create_first_line(self, qfont):
        first_line = QHBoxLayout()
        # creation of the widgets one after another:
        title_label = QLabel("Incident angles (deg)")
        title_label.setFont(qfont)
        informative_label = QLabel("1 angle per line, respecting the stack order")
        informative_label.setStyleSheet(f"color: gray; font-style: italic; font-size: {settings.FontSize.SMALL}pt;")
        # build the widgets together to make the first line:
        first_line.addWidget(title_label)
        first_line.addStretch()
        first_line.addWidget(informative_label)
        return first_line

    def create_button(self):
        button = QPushButton()
        button.setText("Save modifications" if self.parent.is_file_selected else "Create file")
        button.setFont(QFont("", settings.FontSize.BIG))
        button.setSizePolicy(button.sizePolicy().horizontalPolicy(), button.sizePolicy().verticalPolicy())
        button.setStyleSheet("padding: 6px 12px;")
        if self.parent.is_file_selected:
            button.clicked.connect(self.save_json_file)
        else:
            button.clicked.connect(self.create_file)
        # assemble the button in a layout:
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(button)
        layout.addStretch()
        # wrap the layout in a widget
        container = QWidget()
        container.setLayout(layout)
        return container

    def save_json_file(self):
        data = self.collect_current_parameters(when_saving=True)
        if data != 'error':
            save_json(data, self.json_path)
            self.close()

    def create_file(self):
        save_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Save Parameters File",
                                                   str(MEASUREMENTS_DIR),
                                                   "JSON Files (*.json)")
        if save_path:
            self.json_path = save_path
            self.save_json_file()
            self.parent.update_selected_file(save_path)

    def collect_current_parameters(self, when_saving=False):
        error_message = ""
        try:
            params = {"angles_deg": [float(angle.strip(',')) for angle in self.findChild(QTextEdit).toPlainText().splitlines() if
                                     angle.strip()]}
        except ValueError as e:
            params = {}
            error_message += "Cannot save the modifications:\n\n"
            error_message += f"- {type(e).__name__}: {e}.\n"\
                             "Please check any characters that would cause an anomaly in the incident angles (for "\
                             "example a ',' instead of a '.').\n\n"
        for param_name, widget in self.parameter_widgets.items():
            params[param_name] = widget.param_value
            if widget.param_value is None:
                if error_message == "": error_message += "Cannot save the modifications:\n\n"
                error_message += f"- Parameter '{param_name}' has no valid value.\n"
        if error_message != "" and when_saving:
            QMessageBox.warning(
                self,
                "", error_message
            )
            return 'error'
        return params

    def update_ui_current_parameters(self):
        params = self.collect_current_parameters()
        for param_name, param_value in params.items():
            if param_name != 'angles_deg':
                widget = self.parameter_widgets[param_name]
                widget.update_callback()

    def closeEvent(self, event):
        if self.json_path is not None:
            self.parent.json_path = self.json_path
            self.parent.is_file_selected = True
        self.parent.measurement_parameters_editor = None
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # escape is clicked
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)

