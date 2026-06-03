from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox

from .psf_parameters_ui_dictionary import PSF_PARAMETERS_UI
from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator
from deconv.in_out import load_json, save_json, DECONV_MEASUREMENTS_DIR
import settings


## editor window to create or modify a PSF parameters JSON file:
class PsfParametersEditor(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.params_ui_dict = PSF_PARAMETERS_UI
        self.parameter_widgets = {}
        self.parent = parent
        self.json_path = self.parent.json_path if self.parent.is_file_selected else None
        self.json_file = load_json(self.json_path) if self.parent.is_file_selected else None
        self.setWindowTitle("Modify PSF Parameters" if self.parent.is_file_selected else "Create PSF Parameters")
        self.resize(600, 1)
        self._setup_ui()
        self._update_ui_current_parameters()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 10)
        layout.setSpacing(3)
        first = True
        for param_name, param_config in self.params_ui_dict.items():
            if not first:
                layout.addWidget(QSeparator('H'))
            if self.parent.is_file_selected:
                param_config["param_info"]["default"] = self.json_file[param_name]
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"],
            )
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first = False
        layout.addWidget(self._create_button())
        self.setLayout(layout)

    def _create_button(self):
        button = QPushButton()
        button.setText("Save modifications" if self.parent.is_file_selected else "Create file")
        button.setFont(QFont("", settings.FontSize.BIG))
        button.setStyleSheet("padding: 6px 12px;")
        if self.parent.is_file_selected:
            button.clicked.connect(self._save_json)
        else:
            button.clicked.connect(self._create_file)
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(button)
        layout.addStretch()
        container = QWidget()
        container.setLayout(layout)
        return container

    def _save_json(self):
        data = self._collect_parameters(when_saving=True)
        if data != 'error':
            save_json(data, self.json_path)
            self.close()

    def _create_file(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save PSF Parameters File", str(DECONV_MEASUREMENTS_DIR), "JSON Files (*.json)")
        if save_path:
            self.json_path = save_path
            self._save_json()
            self.parent.update_selected_file(save_path)

    def _collect_parameters(self, when_saving=False):
        error_message = ""
        params = {}
        for param_name, widget in self.parameter_widgets.items():
            params[param_name] = widget.param_value
            if widget.param_value is None:
                if error_message == "":
                    error_message += "Cannot save the modifications:\n\n"
                error_message += f"- Parameter '{param_name}' has no valid value.\n"
        if error_message != "" and when_saving:
            QMessageBox.warning(self, "", error_message)
            return 'error'
        return params

    def _update_ui_current_parameters(self):
        for param_name, widget in self.parameter_widgets.items():
            widget.update_callback()

    def closeEvent(self, event):
        if self.json_path is not None:
            self.parent.json_path = self.json_path
            self.parent.is_file_selected = True
        self.parent.psf_parameters_editor = None
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)
