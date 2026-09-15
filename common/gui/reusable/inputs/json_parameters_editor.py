"""
Generic editor window to CREATE or MODIFY a JSON parameters file.

Reused by any inverse problem whose input-files section needs a parameters editor
(matirf's measurement parameters, deconv's PSF parameters, ...). It carries the whole
common body:
    > Create vs Modify mode, deduced from the owning selector (parent.is_file_selected):
        - Modify: values are pre-filled from the selected JSON, button "Save modifications"
        - Create: values start at their UI-dict defaults, button "Create file" (asks a path,
                  then registers it on the selector via parent.update_selected_file)
    > one SimpleParameterWidget per entry of a params UI dict
    > parameter collection with a sentinel: an unset ('None') value shows a QMessageBox and
      aborts the save (collect returns the string 'error')
    > closeEvent coupling back to the owning FileSelector (selected_path / is_file_selected /
      sub_window) and Escape-to-defocus.

Problem-specific EXTRA fields that are not plain SimpleParameterWidgets (e.g. matirf's list
of incident angles, entered as free text) are added by overriding two hooks:
    build_extra_widgets()        -> [QWidget, ...]  inserted at the top
    collect_extra_parameters()   -> (params_dict, error_message)
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox

from common.gui.file_dialog import save_file
from common.gui.base import SimpleParameterWidget
from common.gui.widgets import QSeparator
from common.in_out import load_json, save_json
import common.settings as settings


class JsonParametersEditor(QWidget):

    def __init__(self, parent, *, params_ui_dict, measurements_dir,
                 title_noun="Parameters", width=600):
        super().__init__()
        self.parent = parent                 # the owning FileSelector
        self.params_ui_dict = params_ui_dict
        self._measurements_dir = measurements_dir
        self._title_noun = title_noun
        self.parameter_widgets = {}
        self.is_modify = self.parent.is_file_selected
        self.json_path = self.parent.selected_path if self.is_modify else None
        self.json_file = load_json(self.json_path) if self.is_modify else None
        self.setWindowTitle(f"{'Modify' if self.is_modify else 'Create'} {title_noun}")
        self.resize(width, 1)
        self._setup_ui()
        self._update_ui_current_parameters()

    # ── hooks for problem-specific extra fields (default: none) ──────────────

    def build_extra_widgets(self):
        """Widgets to insert at the TOP, before the UI-dict parameters. Override to add some."""
        return []

    def collect_extra_parameters(self):
        """Return (params_dict, error_message). Override alongside build_extra_widgets()."""
        return {}, ""

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 10)
        layout.setSpacing(3)
        # problem-specific extra fields first, each followed by a separator:
        for widget in self.build_extra_widgets():
            layout.addWidget(widget)
            layout.addWidget(QSeparator('H'))
        first = True
        for param_name, param_config in self.params_ui_dict.items():
            if not first:
                layout.addWidget(QSeparator('H'))
            # copy param_info so we never mutate the shared, module-level UI dict when we
            # inject the current file's value as the default in Modify mode:
            param_info = dict(param_config["param_info"])
            if self.is_modify:
                param_info["default"] = self.json_file[param_name]
            widget = SimpleParameterWidget(
                title=param_config["title"], type=param_config["type"], param_info=param_info)
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first = False
        layout.addWidget(self._create_button())
        self.setLayout(layout)

    def _create_button(self):
        button = QPushButton("Save modifications" if self.is_modify else "Create file")
        button.setFont(QFont("", settings.FontSize.BIG))
        button.setStyleSheet("padding: 6px 12px;")
        button.clicked.connect(self._save_json if self.is_modify else self._create_file)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button)
        row.addStretch()
        container = QWidget()
        container.setLayout(row)
        return container

    # ── save / create ────────────────────────────────────────────────────────

    def _save_json(self):
        data = self._collect_parameters(when_saving=True)
        if data != 'error':
            save_json(data, self.json_path)
            self.close()

    def _create_file(self):
        save_path = save_file(self, f"Save {self._title_noun} File",
                              self._measurements_dir, "JSON Files (*.json)")
        if save_path:
            if not save_path.lower().endswith('.json'):
                save_path += '.json'   # the dialog does not force the extension
            self.json_path = save_path
            self._save_json()
            self.parent.update_selected_file(save_path)

    def _collect_parameters(self, when_saving=False):
        params, errors = {}, []
        extra_params, extra_error = self.collect_extra_parameters()
        params.update(extra_params)
        if extra_error:
            errors.append(extra_error)
        for param_name, widget in self.parameter_widgets.items():
            params[param_name] = widget.param_value
            if widget.param_value is None:
                errors.append(f"- Parameter '{param_name}' has no valid value.")
        if errors and when_saving:
            QMessageBox.warning(self, "", "Cannot save the modifications:\n\n" + "\n".join(errors))
            return 'error'
        return params

    def _update_ui_current_parameters(self):
        for widget in self.parameter_widgets.values():
            widget.update_callback()

    # ── close / key events ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.json_path is not None:
            self.parent.selected_path = self.json_path
            self.parent.is_file_selected = True
        self.parent.sub_window = None
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)


if __name__=="__main__":  # test
    import sys
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory

    import common.settings as settings


    class _FakeSelector:
        """Minimal stand-in for the owning FileSelector, here in CREATE mode."""
        is_file_selected = False
        selected_path = None
        sub_window = None
        def update_selected_file(self, path):
            print(f"[demo] file created/selected: {path}")

    DEMO_UI = {
        "sigma": {"title": "A float value", "type": "value",
                  "param_info": {'dtype': float, 'unit': 'px', 'latex_name': '\\sigma', 'default': None}},
        "enabled": {"title": "A boolean", "type": "bool", "param_info": {'default': True}},
        "kind": {"title": "An option", "type": "option", "param_info": {'options_list': ['A', 'B', 'C']}},
    }

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    editor = JsonParametersEditor(
        _FakeSelector(),
        params_ui_dict=DEMO_UI,
        measurements_dir=str(Path(tempfile.mkdtemp())),
        title_noun="Demo Parameters",
    )
    editor.show()
    sys.exit(app.exec_())
