"""
Abstract base class for the input-files section of any inverse problem.

Provides:
    - A real / synthetic mode toggle (QSwitchButton + labels)
    - Two file selectors side-by-side (image + JSON parameters)
    - update_ui_from_toml() to reload the UI from a config file

Hooks to implement:
    _get_cached_mode()                  -> bool  (True = real, False = synthetic)
    _real_mode_text()                   -> str
    _synthetic_mode_text()              -> str
    _create_image_selector()            -> QWidget  (must expose .is_file_selected,
                                                      .update_mode(),
                                                      .update_selected_file(path))
    _create_json_selector()             -> QWidget  (same interface)
    _on_switch_mode(is_mode_real)       -> None  (persist mode to cache)
    _load_config_for_update(toml_path)  -> dict
    _get_file_paths_from_config(config) -> (image_path, json_path)
"""

from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel

from common.gui.widgets import QSwitchButton, QSeparator
from common import DataMode
from common.settings import FontSize


class BaseInputFilesSection(QGroupBox):

    def __init__(self, parent=None):
        super().__init__("Input Files")
        self.on_color = self.palette().color(QPalette.WindowText).name()
        self.off_color = self.palette().color(QPalette.PlaceholderText).name()
        self.parent = parent
        self.is_mode_real = self._get_cached_mode()
        self._setup_ui()

    # ── hooks (to override) ─────────────────────────────────────────────

    def _get_cached_mode(self) -> bool:
        raise NotImplementedError

    def _real_mode_text(self) -> str:
        raise NotImplementedError

    def _synthetic_mode_text(self) -> str:
        raise NotImplementedError

    def _create_image_selector(self):
        raise NotImplementedError

    def _create_json_selector(self):
        raise NotImplementedError

    def _on_switch_mode(self, is_mode_real: bool):
        """Called after the toggle.  Persist the new mode to cache."""
        raise NotImplementedError

    def _load_config_for_update(self, toml_path) -> dict:
        raise NotImplementedError

    def _get_file_paths_from_config(self, config) -> tuple:
        """Return (image_file_path, json_file_path) from the config dict."""
        raise NotImplementedError

    # ── shared UI construction ──────────────────────────────────────────

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
        self.real_label = QLabel(self._real_mode_text())
        self.synth_label = QLabel(self._synthetic_mode_text())
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
        self.image_selector = self._create_image_selector()
        self.json_selector = self._create_json_selector()
        bot.addWidget(self.image_selector)
        bot.addWidget(QSeparator('V'))
        bot.addWidget(self.json_selector)
        return bot

    # ── shared logic ────────────────────────────────────────────────────

    def _update_labels(self):
        on = f"color: {self.on_color}; font-style: italic; font-size: {FontSize.SMALL}pt;"
        off = f"color: {self.off_color}; font-style: italic; font-size: {FontSize.SMALL}pt;"
        self.real_label.setStyleSheet(on if self.is_mode_real else off)
        self.synth_label.setStyleSheet(off if self.is_mode_real else on)

    def _switch_mode(self):
        self.is_mode_real = not self.is_mode_real
        self._update_labels()
        self._on_switch_mode(self.is_mode_real)
        self.image_selector.update_mode()
        self.json_selector.update_mode()

    def are_both_file_selected(self):
        return self.image_selector.is_file_selected and self.json_selector.is_file_selected

    def update_ui_from_toml(self, toml_path):
        config = self._load_config_for_update(toml_path)
        mode = config['input-paths']['mode']
        self.is_mode_real = (mode == DataMode.REAL.value)
        self._update_labels()
        self.image_selector.update_mode()
        self.json_selector.update_mode()
        if self.is_mode_real:
            self.switch_button.switch_to_left(no_signal=True)
        else:
            self.switch_button.switch_to_right(no_signal=True)
        image_path, json_path = self._get_file_paths_from_config(config)
        self.image_selector.update_selected_file(image_path)
        self.json_selector.update_selected_file(json_path)
