"""
Standalone settings GUI — launched by ``settings gui``.

A small QDialog with one widget per setting, plus Apply/Reset buttons.
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyleFactory,
    QVBoxLayout,
)

from .settings import Settings, _DEFAULTS


class SettingsDialog(QDialog):

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._widgets = {}

        self.setWindowTitle("matirf_python_plus — Settings")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        for key, meta in _DEFAULTS.items():
            widget = self._make_widget(key, meta)
            self._widgets[key] = widget
            label = QLabel(meta["description"])
            label.setToolTip(key)
            form.addRow(label, widget)
        layout.addLayout(form)

        buttons = QHBoxLayout()

        btn_apply = QPushButton("Apply && Save")
        btn_apply.clicked.connect(self._apply)
        buttons.addWidget(btn_apply)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._reset)
        buttons.addWidget(btn_reset)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        buttons.addWidget(btn_close)

        layout.addLayout(buttons)

    def _make_widget(self, key, meta):
        value = self._settings.get(key)
        choices = meta.get("choices")

        if isinstance(meta["value"], bool):
            w = QCheckBox()
            w.setChecked(value)
            return w

        if choices:
            w = QComboBox()
            w.addItems(choices)
            w.setCurrentText(str(value))
            return w

        if isinstance(meta["value"], int):
            w = QSpinBox()
            w.setRange(1, 9999)
            w.setValue(value)
            return w

        w = QComboBox()
        w.setEditable(True)
        w.setCurrentText(str(value))
        return w

    def _read_widget(self, key):
        w = self._widgets[key]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QSpinBox):
            return w.value()
        if isinstance(w, QComboBox):
            return w.currentText()
        return str(w.text())

    def _apply(self):
        for key in _DEFAULTS:
            value = self._read_widget(key)
            self._settings.set(key, value)

    def _reset(self):
        self._settings.reset()
        for key, meta in _DEFAULTS.items():
            w = self._widgets[key]
            value = meta["value"]
            if isinstance(w, QCheckBox):
                w.setChecked(value)
            elif isinstance(w, QSpinBox):
                w.setValue(value)
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(value))


def open_settings_gui():
    from common.settings import _settings

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(_settings.get("app_style")))

    from common.settings import dark_palette, light_palette
    palette = dark_palette if _settings.get("dark_style") else light_palette
    app.setPalette(palette())

    dialog = SettingsDialog(_settings)
    dialog.setWindowFlag(Qt.WindowStaysOnTopHint, False)
    dialog.show()
    sys.exit(app.exec_())
