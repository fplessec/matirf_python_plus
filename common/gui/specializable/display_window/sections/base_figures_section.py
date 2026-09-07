"""
Abstract base class for the figures section of any display window.

Provides the full UI skeleton: optional multi-view switching (header with
toggle button and labels) and a stacked layout to hold the views.

Hooks to implement:
    create_views()                -> list[QWidget]   (one widget per view)
    update_views(f, config)       -> None             (update existing viewer data in-place)
    view_labels()                 -> list[str]        (optional, defaults to "view 1", "view 2", ...)
"""

from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QStackedLayout, QWidget, QVBoxLayout, QPushButton,
)

from common.gui.widgets import QSwitchButton
from common.gui.file_dialog import save_file
import common.settings as settings


class BaseFiguresSection(QGroupBox):
    """
    An object that manages one or multiple switchable views for displaying
    reconstruction results. If only one view is returned by create_views(),
    no switch header is shown.
    """

    def __init__(self, parent=None):
        super().__init__("Figures")
        self.parent = parent
        self._current_view_index = 0
        self._views = []
        self._labels = []
        self._initialized = False
        self._setup_ui()

    # ── hooks (to override) ──────────────────────────────────────────────

    def create_views(self):
        """Returns a list of QWidget, one per view. Must be overridden by subclasses."""
        raise NotImplementedError

    def update_views(self, f, config):
        """Updates the existing viewers in-place with new data. Must be overridden by subclasses."""
        raise NotImplementedError

    def view_labels(self):
        """Returns a list of labels for each view. Default: 'view 1', 'view 2', ..."""
        return [f"view {i + 1}" for i in range(len(self._views))]

    # ── view-1 PNG export hooks (override in subclasses that support it) ──────

    def supports_view1_export(self):
        """Whether a 'save view 1 as PNG' button should appear. Override to enable."""
        return False

    def export_view1(self, filepath):
        """Saves view 1 (images + legends) as a PNG. Override when supported."""
        raise NotImplementedError

    def export_default_dir(self):
        """Directory the save dialog opens at. Override to point somewhere sensible."""
        from pathlib import Path
        return str(Path.home())

    # ── UI construction ──────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        ## stacked layout to hold the views (populated on first update_plot call):
        self._stacked_layout = QStackedLayout()
        ## placeholder for the header (created only if multiple views):
        self._header = None
        layout.addLayout(self._stacked_layout)
        self.setLayout(layout)

    def _create_header(self):
        """Creates the switch header with labels and toggle button (only for multi-view)."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setSpacing(6)
        ## creation of the widgets one after another:
        labels = self.view_labels()
        self._label_widgets = []
        self._label_widgets.append(QLabel(labels[0]))
        self._label_widgets.append(QLabel(labels[1]))
        self._switch = QSwitchButton()
        self._switch.toggled.connect(self._on_switch_toggled)
        ## leftmost: 'save view 1 as PNG' button (shown only in view 1, if supported):
        self._export_btn = QPushButton("Save view as PNG")
        self._export_btn.setToolTip("Save this view (depth map + profiles) as a PNG image")
        self._export_btn.clicked.connect(self._on_export_view1)
        ## layout: 'Save view as PNG' far left | stretch | switch group far right
        layout.addWidget(self._export_btn)
        layout.addStretch()
        layout.addWidget(self._label_widgets[0])
        layout.addWidget(self._switch)
        layout.addWidget(self._label_widgets[1])
        self._update_label_styles()
        self._update_export_button_visibility()
        return header

    def _update_export_button_visibility(self):
        """The export button is visible only in view 1 and only when export is supported."""
        if hasattr(self, '_export_btn'):
            self._export_btn.setVisible(
                self.supports_view1_export() and self._current_view_index == 0)

    def _on_export_view1(self):
        path = save_file(self, "Save view as PNG", self.export_default_dir(), "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith('.png'):
            path += '.png'
        self.export_view1(path)

    # ── public API ───────────────────────────────────────────────────────

    def update_plot(self, f, config):
        """Called to display or refresh the reconstruction data in the viewers."""
        ## update_views is called first so that subclasses can store data (f, config params)
        ## needed by create_views() on the initial call:
        self.update_views(f, config)
        if not self._initialized:
            self._initialize_views()
            self._initialized = True

    # ── internals ────────────────────────────────────────────────────────

    def _initialize_views(self):
        """Creates the views and the switch header on the first update_plot call."""
        self._views = self.create_views()
        ## add each view to the stacked layout:
        for view in self._views:
            self._stacked_layout.addWidget(view)
        ## if multiple views, insert the header above the stacked layout:
        if len(self._views) > 1:
            self._header = self._create_header()
            main_layout = self.layout()
            main_layout.insertWidget(0, self._header)

    def _on_switch_toggled(self):
        self._current_view_index = 0 if self._current_view_index == 1 else 1
        self._stacked_layout.setCurrentIndex(self._current_view_index)
        self._update_label_styles()
        self._update_export_button_visibility()

    def _update_label_styles(self):
        """Highlights the active view label and grays out the inactive one."""
        if not hasattr(self, '_label_widgets'):
            return
        for i, label in enumerate(self._label_widgets):
            if i == self._current_view_index:
                label.setStyleSheet(f"color: white; font-size: {settings.FontSize.SMALL}pt;")
            else:
                label.setStyleSheet(f"color: gray; font-size: {settings.FontSize.SMALL}pt;")
