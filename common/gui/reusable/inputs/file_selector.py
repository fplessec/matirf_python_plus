"""
Generic file-path selector, reused by any inverse problem's input-files section.

A single configurable widget replaces the near-identical per-problem selectors
(matirf's Tif/Json selectors, deconv's Png/Json selectors). It carries the whole common
body — mode-dependent title, "Choose ..." button, an optional problem-specific secondary
button, the selected-file label, and a cross button to unselect — and takes only:
    > locations   : the measurements directory, and the TOML key to write under 'input-paths'
    > problem bits: file extension + dialog filter, the two mode-dependent titles, an optional
                    file validator, and an optional SelectorButton (a secondary button whose
                    click opens a problem-specific sub-window: a preview viewer, an editor, ...)

State handling (the subtle part) is centralized in the section:
    > the title depends on the mode (real/synthetic)               -> refreshed by update_mode()
    > the file label depends on this selector's selection          -> refreshed on (un)select
    > a secondary button's text/visibility may depend on the WHOLE section state (e.g. "show
      the preview button only when BOTH files are selected") -> after any (un)select the
      selector calls section.notify_selection_changed(), which re-evaluates every selector's
      secondary button. No selector reaches into another selector's widgets.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional, Union

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

from common.gui.file_dialog import open_file
from common.gui.widgets import QCrossButton
from common.settings import FontSize


@dataclass
class SelectorButton:
    """
    A problem-specific secondary button on a FileSelector.

    on_click     : callable(selector) -> None. Receives the FileSelector so the problem can
                   build/show its own sub-window (via selector.set_sub_window(...)) and handle
                   its own errors.
    text         : a static label, OR a callable(is_file_selected: bool) -> str for a label
                   that follows the selection (e.g. "Create" / "Modify").
    visible_when : callable() -> bool, re-evaluated on every selection change across the whole
                   section (default: always visible). E.g. section.are_both_file_selected.
    """
    on_click: Callable
    text: Union[str, Callable] = ""
    visible_when: Optional[Callable] = None


class FileSelector(QWidget):

    def __init__(self, section, *, noun, dialog_filter, toml_key, measurements_dir,
                 update_cache_fn, title_real, title_synthetic,
                 validate_fn=None, extra_button=None):
        super().__init__()
        # 'section' is the BaseInputFilesSection: exposes is_mode_real,
        # are_both_file_selected(), notify_selection_changed().
        self.parent = section
        self.is_file_selected = False
        self.selected_path = None
        self.sub_window = None            # the problem-specific sub-window (viewer/editor), if open
        self._noun = noun
        self._dialog_filter = dialog_filter
        self._toml_key = toml_key
        self._measurements_dir = measurements_dir
        self._update_cache = update_cache_fn
        self._title_real = title_real
        self._title_synthetic = title_synthetic
        self._validate = validate_fn
        self._extra = extra_button
        self._setup_ui()

    # ── UI construction ──────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(10)
        self.title_label = QLabel(self._title_text())
        self.title_label.setStyleSheet(f"font-size: {FontSize.NORMAL}pt;")
        choose_btn = QPushButton(f"Choose .{self._noun} file")
        choose_btn.clicked.connect(self._choose_file)
        self._extra_btn = None
        if self._extra is not None:
            self._extra_btn = QPushButton()
            self._extra_btn.clicked.connect(lambda: self._extra.on_click(self))
        self.file_label = QLabel(self._no_file_text())
        color = self.palette().color(QPalette.PlaceholderText).name()
        self.file_label.setStyleSheet(
            f"color: {color}; font-style: italic; font-size: {FontSize.NORMAL}pt;")
        unselect_btn = QCrossButton()
        unselect_btn.clicked.connect(self._unselect)
        last_line = QHBoxLayout()
        layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addWidget(choose_btn, alignment=Qt.AlignHCenter)
        if self._extra_btn is not None:
            layout.addWidget(self._extra_btn, alignment=Qt.AlignHCenter)
        layout.addStretch()
        last_line.addStretch()
        last_line.addWidget(self.file_label, alignment=Qt.AlignHCenter)
        last_line.addWidget(unselect_btn)
        last_line.addStretch()
        layout.addLayout(last_line)
        self.setLayout(layout)
        self.refresh_extra_button()

    # ── text helpers ───────────────────────────────────────────────────────

    def _title_text(self):
        return self._title_real if self.parent.is_mode_real else self._title_synthetic

    def _no_file_text(self):
        return f"No .{self._noun} file selected"

    # ── actions ────────────────────────────────────────────────────────────

    def _choose_file(self):
        path = open_file(self, self._title_text(), self._measurements_dir, self._dialog_filter)
        if not path:
            return
        if self._validate is not None and not self._validate(path):
            return
        self.update_selected_file(path)

    def update_selected_file(self, path):
        self.selected_path = path
        self._update_cache(["input-paths", self._toml_key], path)
        self.is_file_selected = (path != 'None')
        self._refresh_file_label(os.path.basename(path))
        self.parent.notify_selection_changed()

    def _unselect(self):
        self._update_cache(["input-paths", self._toml_key], 'None')
        self.selected_path = None
        self.is_file_selected = False
        self._refresh_file_label('None')
        self.parent.notify_selection_changed()

    # ── sub-window (problem-specific viewer/editor) ─────────────────────────

    def set_sub_window(self, window):
        """Store and show a problem-specific sub-window, closing any previous one."""
        self.close_sub_window()
        self.sub_window = window
        if window is not None:
            window.show()

    def close_sub_window(self):
        if self.sub_window is not None:
            self.sub_window.close()
            self.sub_window = None

    # ── state -> UI ─────────────────────────────────────────────────────────

    def _refresh_file_label(self, file_name):
        self.file_label.setText(
            f"selected file : {file_name}" if self.is_file_selected else self._no_file_text())
        font = self.file_label.font()
        font.setBold(self.is_file_selected)
        self.file_label.setFont(font)

    def refresh_extra_button(self):
        """Re-evaluate the secondary button's text + visibility from the current section state."""
        if self._extra_btn is None:
            return
        text = self._extra.text
        self._extra_btn.setText(text(self.is_file_selected) if callable(text) else text)
        if self._extra.visible_when is None:
            visible = True
        else:
            try:
                visible = bool(self._extra.visible_when())
            except Exception:
                # e.g. evaluated before both selectors exist -> hide until the section refreshes
                visible = False
        self._extra_btn.setVisible(visible)

    def update_mode(self):
        """Called by the section on a mode switch / TOML reload: refresh the mode-dependent title."""
        self.title_label.setText(self._title_text())


if __name__=="__main__":  # test
    import sys
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory, QGroupBox, QHBoxLayout

    import common.settings as settings
    from common.cache import make_update_cache
    from common.in_out import load_or_create_toml


    class _DemoSection(QGroupBox):
        """Minimal stand-in for BaseInputFilesSection: two selectors + the notify hook that
        re-evaluates their state-dependent secondary buttons (so the 'preview' button appears
        only once BOTH files are selected)."""
        is_mode_real = True

        def __init__(self, update_cache):
            super().__init__("test of object: FileSelector — select BOTH files to reveal 'preview'")
            self.image_selector = FileSelector(
                self, noun="tif", dialog_filter="Image Files (*.tif *.tiff)", toml_key="tif",
                measurements_dir=str(Path.home()), update_cache_fn=update_cache,
                title_real="Path of the image", title_synthetic="Path of the truth",
                extra_button=SelectorButton(
                    text="See preprocessed file",
                    on_click=lambda sel: print("[demo] preview clicked"),
                    visible_when=self.are_both_file_selected))
            self.json_selector = FileSelector(
                self, noun="json", dialog_filter="Parameters Files (*.json)", toml_key="json",
                measurements_dir=str(Path.home()), update_cache_fn=update_cache,
                title_real="Path of the parameters", title_synthetic="Path of the sim parameters",
                extra_button=SelectorButton(
                    text=lambda selected: "Modify .json file" if selected else "Create .json file",
                    on_click=lambda sel: print("[demo] create/modify clicked")))
            row = QHBoxLayout(self)
            row.addWidget(self.image_selector)
            row.addWidget(self.json_selector)
            self.notify_selection_changed()

        def are_both_file_selected(self):
            return self.image_selector.is_file_selected and self.json_selector.is_file_selected

        def notify_selection_changed(self):
            self.image_selector.refresh_extra_button()
            self.json_selector.refresh_extra_button()


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    # a throwaway TOML so the selectors' cache-sync path is exercised (never the real cache):
    tmp_toml = Path(tempfile.mkdtemp()) / "demo_cache.toml"
    update_cache = make_update_cache(tmp_toml, {"input-paths": {"tif": "None", "json": "None"}})
    window = _DemoSection(update_cache)
    window.resize(720, 320)
    window.show()
    sys.exit(app.exec_())
