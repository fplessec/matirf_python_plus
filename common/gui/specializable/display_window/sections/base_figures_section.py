"""
Abstract base class for the figures section of any display window.

Provides the full UI skeleton: optional multi-view switching (header with
toggle button and labels) and a stacked layout to hold the views.

Hooks to implement:
    create_views()                -> list[QWidget]   (one widget per view)
    update_views(f, config)       -> None             (update existing viewer data in-place)
    view_labels()                 -> list[str]        (optional, defaults to "view 1", "view 2", ...)
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QStackedLayout, QWidget, QVBoxLayout

from common.gui.widgets import QSwitchButton
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
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(6)
        ## creation of the widgets one after another:
        labels = self.view_labels()
        self._label_widgets = []
        self._label_widgets.append(QLabel(labels[0]))
        self._label_widgets.append(QLabel(labels[1]))
        self._switch = QSwitchButton()
        self._switch.toggled.connect(self._on_switch_toggled)
        ## build the widgets together to make the layout:
        layout.addWidget(self._label_widgets[0])
        layout.addWidget(self._switch)
        layout.addWidget(self._label_widgets[1])
        layout.addStretch()
        self._update_label_styles()
        return header

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

    def _update_label_styles(self):
        """Highlights the active view label and grays out the inactive one."""
        if not hasattr(self, '_label_widgets'):
            return
        for i, label in enumerate(self._label_widgets):
            if i == self._current_view_index:
                label.setStyleSheet(f"color: white; font-size: {settings.FontSize.SMALL}pt;")
            else:
                label.setStyleSheet(f"color: gray; font-size: {settings.FontSize.SMALL}pt;")
