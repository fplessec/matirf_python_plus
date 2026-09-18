"""
Base figures section for any display window.

A concrete figures section is declared by a `VIEWS` spec (list of `View`), like the other
GUI layers are declared by dictionaries/attributes. The base handles everything else:
multi-view switching (header + toggle), the stacked layout, lazy view creation, in-place
refresh of the active view, and the optional per-view PNG export.

Declarative model:
    VIEWS = [ View(label, panels=[Panel(...), ...], orientation="horizontal"|"vertical",
                   export=optional callable(view_panel_widgets, filepath)), ... ]
    each Panel wires ONE viewer widget:
        build(f, config)          -> the widget (created lazily on the first update)
        refresh(widget, f, config)-> update it in place
    For the common case (a viewer built from the image, refreshed via set_image), use the
    `image_panel(viewer_class, title=...)` helper.

Optional class attribute:
    EXPORT_DEFAULT_DIR   directory the PNG-export dialog opens at (a path/str)

Subclasses may still override create_views()/update_views() directly for fully custom
behaviour; the VIEWS-driven implementations below are the defaults.
"""

from dataclasses import dataclass
from typing import Callable, List

from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QStackedLayout, QWidget, QVBoxLayout, QPushButton,
)

from common.gui.widgets import QSwitchButton
from common.gui.file_dialog import save_file
import common.settings as settings


@dataclass
class Panel:
    """
    One viewer widget inside a view.

    >> build   : callable    (f, config) -> QWidget   (create the viewer, lazily)
    >> refresh : callable    (widget, f, config) -> None   (update it in place)
    >> stretch : int         layout stretch factor within the view (default 1)

    For the common "image viewer refreshed via set_image(f)" case, use image_panel() instead
    of writing build/refresh by hand.
    """
    build: Callable
    refresh: Callable
    stretch: int = 1


@dataclass
class View:
    """
    A single switchable view = a row/column of panels, with an optional PNG export.

    >> label       : str          shown in the switch header
    >> panels      : list[Panel]  the viewers laid out in this view
    >> orientation : str          'horizontal' (default) | 'vertical'
    >> export      : callable     optional (view_panel_widgets, filepath) -> None: enables the
                                  "Save view as PNG" button; view_panel_widgets is the list of
                                  this view's built panel widgets, in order
    """
    label: str
    panels: List[Panel]
    orientation: str = "horizontal"
    export: Callable = None


def image_panel(viewer_class, title="", stretch=1):
    """Panel for the common case: a viewer built from the image and refreshed via set_image(f)."""
    return Panel(
        build=lambda f, config: viewer_class(f, title=title),
        refresh=lambda widget, f, config: widget.set_image(f),
        stretch=stretch,
    )


class BaseFiguresSection(QGroupBox):
    """
    Figures section of a display window — declared by a VIEWS spec.

    Manages one or several switchable views of the reconstruction. A single view shows no
    switch header; several views add a header with a toggle (and the optional PNG export).

    ----------
    > Parameters (override as class attributes) :
    ----------

    >> VIEWS : list[View]
        The views to display. Each View holds Panels (one viewer widget each). See
        View / Panel / image_panel above.

    >> EXPORT_DEFAULT_DIR : Path or str
        Directory the "Save view as PNG" dialog opens in (only used if a View has export=...).

    ----------
    > Example :
    ----------

        class FiguresSection(BaseFiguresSection):
            EXPORT_DEFAULT_DIR = MATIRF_RESULTS_DIR
            VIEWS = [
                View("view 1",
                     [Panel(build_depth_map, refresh_z_view, stretch=2),   # needs z0/zN from config
                      Panel(build_profiles, refresh_z_view, stretch=1)],
                     export=export_view1_fn),
                View("view 2",
                     [image_panel(ImageAndHisto3DViewer, title="")]),      # common set_image(f) case
            ]

    Subclasses may still override create_views()/update_views() directly for fully custom
    behaviour; the VIEWS-driven implementations are only the defaults.
    """

    VIEWS = []                 # declarative spec (list[View]); drives the default hooks below
    EXPORT_DEFAULT_DIR = None  # directory the PNG-export dialog opens at

    def __init__(self, parent=None):
        super().__init__("Figures")
        self.parent = parent
        self._current_view_index = 0
        self._views = []
        self._panel_widgets = []   # per view: list of the built panel widgets
        self._f = None
        self._config = None
        self._initialized = False
        self._setup_ui()

    # ── hooks (VIEWS-driven defaults; override for fully custom behaviour) ────

    def create_views(self):
        """Build one QWidget per View in VIEWS (lazy, on the first update)."""
        self._panel_widgets = []
        views = []
        for view in self.VIEWS:
            container = QWidget()
            box = (QHBoxLayout if view.orientation == "horizontal" else QVBoxLayout)(container)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(0)
            widgets = []
            for panel in view.panels:
                widget = panel.build(self._f, self._config)
                box.addWidget(widget, panel.stretch)
                widgets.append(widget)
            self._panel_widgets.append(widgets)
            views.append(container)
        return views

    def update_views(self, f, config):
        """Store the data, then refresh the ACTIVE view's panels in place."""
        self._f, self._config = f, config
        if self._panel_widgets:
            i = self._current_view_index
            for panel, widget in zip(self.VIEWS[i].panels, self._panel_widgets[i]):
                panel.refresh(widget, f, config)

    def view_labels(self):
        """Labels for each view. Default: the VIEWS labels, else 'view 1', 'view 2', ..."""
        if self.VIEWS:
            return [v.label for v in self.VIEWS]
        return [f"view {i + 1}" for i in range(len(self._views))]

    # ── PNG export (driven by VIEWS[0].export) ───────────────────────────────

    def supports_view1_export(self):
        return bool(self.VIEWS) and self.VIEWS[0].export is not None

    def export_view1(self, filepath):
        self.VIEWS[0].export(self._panel_widgets[0], filepath)

    def export_default_dir(self):
        if self.EXPORT_DEFAULT_DIR is not None:
            return str(self.EXPORT_DEFAULT_DIR)
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
        ## leftmost: 'save view as PNG' button (shown only in view 1, if supported):
        self._export_btn = QPushButton("Save view as PNG")
        self._export_btn.setToolTip("Save this view as a PNG image")
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
        ## refresh the newly-active view so it reflects the latest data:
        if self._panel_widgets and self._f is not None:
            for panel, widget in zip(self.VIEWS[self._current_view_index].panels,
                                     self._panel_widgets[self._current_view_index]):
                panel.refresh(widget, self._f, self._config)
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
