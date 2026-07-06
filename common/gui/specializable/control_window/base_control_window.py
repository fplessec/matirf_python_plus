"""
Declarative base class for the control window of any inverse problem.

Subclasses configure the window entirely through class attributes — no methods
to override (except on_close_cleanup for problem-specific sub-windows).

Section descriptor formats:
    (title, ui_dict, toml_key)  — creates a BaseSectionQGroup from a UI dict
    SomeClass                   — auto-instantiated with injected kwargs
    (SomeClass, extra_kwargs)   — same, with extra kwargs merged in
"""

import re
import inspect

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QAction,
)

from common.gui.file_dialog import open_file, save_file, open_directory
from common.gui.utils import close_active_window
from common.gui.base.base_section_qgroup import BaseSectionQGroup
from common.in_out import load_or_create_toml, save_toml
from common.cache import make_update_cache
import common.settings as settings


def _camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class BaseControlWindow(QMainWindow):

    # ── class attributes (override in subclasses) ───────────────────────

    window_title = ""
    cached_config_path = None       # Path to the cached config.toml
    default_config = None           # dict used as fallback when config.toml does not exist
    results_dir = None              # directory where reconstruction results are stored
    pipeline_class = None   # the BasePipeline subclass for this problem
    display_window_manager_class = None  # the DisplayWindowManager class for this problem

    # Section descriptors: each item is either a (title, ui_dict, toml_key)
    # tuple, a class, or a (class, extra_kwargs) tuple.  See module docstring.
    sections_left = []
    sections_right = []

    # ── init ────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self.update_cache_fn = make_update_cache(self.cached_config_path, self.default_config)
        self.setWindowTitle(self.window_title)
        self.setGeometry(100, 100, settings.width_cw, settings.height_cw)
        self._all_sections = []     # flat list of every built section (for load_cached_config)
        self._setup_ui()
        self.load_cached_config()

    # ── config I/O ─────────────────────────────────────────────────────────

    def load_config(self, path):
        return load_or_create_toml(path, self.default_config)

    def save_config(self, config, path):
        save_toml(config, path)

    # ── section building ────────────────────────────────────────────────

    ## builds a vertical layout from a list of section descriptors
    def _build_column(self, descriptors):
        layout = QVBoxLayout()
        for desc in descriptors:
            section, attr_name = self._build_section(desc)
            layout.addWidget(section)
            self._all_sections.append(section)
            setattr(self, attr_name, section)
        layout.addStretch()
        return layout

    ## builds a single section widget from a descriptor, returns (widget, attr_name)
    def _build_section(self, desc):
        # ── UI-dict tuple: (title, ui_dict, toml_key) ──
        if isinstance(desc, tuple) and len(desc) == 3 and isinstance(desc[1], dict):
            title, ui_dict, toml_key = desc
            section = BaseSectionQGroup(
                parent=self,
                update_cache_fn=self.update_cache_fn,
                load_toml_fn=self.load_config,
                title=title,
                params_ui_dict=ui_dict,
                toml_section_key=toml_key,
            )
            # Attribute name from toml_key: 'add-noise' → 'add_noise'
            attr_name = toml_key.replace('-', '_')
            return section, attr_name
        # ── Class with extra kwargs: (Class, kwargs_dict) ──
        if isinstance(desc, tuple) and len(desc) == 2:
            cls, extra_kwargs = desc
            section = self._instantiate_section(cls, extra_kwargs)
            # Attribute name from class name: AlgorithmSelectionSection → algorithm_selection_section
            attr_name = _camel_to_snake(cls.__name__)
            return section, attr_name
        # ── Plain class ──
        if isinstance(desc, type):
            section = self._instantiate_section(desc, {})
            attr_name = _camel_to_snake(desc.__name__)
            return section, attr_name
        raise TypeError(f"Unknown section descriptor: {desc!r}")

    ## auto-injects well-known kwargs (parent, update_cache_fn, etc.) filtered by constructor signature
    def _instantiate_section(self, cls, extra_kwargs):
        # Standard kwargs that BaseControlWindow can provide to any section
        auto_kwargs = {
            'parent': self,
            'update_cache_fn': self.update_cache_fn,
            'load_toml_fn': self.load_config,
            'config_path': self.cached_config_path,
            'save_toml_fn': self.save_config,
        }
        # extra_kwargs (from the descriptor) override auto_kwargs
        all_kwargs = {**auto_kwargs, **extra_kwargs}
        # Inspect the constructor to find which kwargs it accepts
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters.keys()) - {'self'}
        # If the constructor has **kwargs, pass everything
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_var_kw:
            return cls(**all_kwargs)
        # Otherwise, only pass kwargs that match a declared parameter
        return cls(**{k: v for k, v in all_kwargs.items() if k in accepted})

    # ── shared UI construction ──────────────────────────────────────────

    def _setup_ui(self):
        self.setFocus()
        self._create_menu_bar()
        central = QWidget()
        main_layout = QVBoxLayout()
        columns = QHBoxLayout()
        columns.addLayout(self._build_column(self.sections_left))
        columns.addLayout(self._build_column(self.sections_right))
        main_layout.addLayout(columns)
        main_layout.addStretch()
        main_layout.addWidget(self._create_bottom_bar())
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _create_menu_bar(self):
        menubar = self.menuBar()
        # file menu:
        file_menu = menubar.addMenu('File')
        load_any = QAction(QIcon(), "Load any config", self)
        load_any.setShortcut("Ctrl+L")
        load_any.triggered.connect(self._load_any_config)
        file_menu.addAction(load_any)
        save_action = QAction(QIcon(), "Save config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)
        # reconstruction menu:
        recon_menu = menubar.addMenu('Reconstruction')
        run_action = QAction(QIcon(), "Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self._run)
        recon_menu.addAction(run_action)
        open_action = QAction(QIcon(), "Open reconstruction", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_reconstruction)
        recon_menu.addAction(open_action)
        # window menu:
        window_menu = menubar.addMenu('Window')
        close_this = QAction(QIcon(), "Close this window", self)
        close_this.setShortcut("Ctrl+W")
        close_this.triggered.connect(close_active_window)
        window_menu.addAction(close_this)
        close_all = QAction(QIcon(), "Close all reconstruction windows", self)
        close_all.setShortcut("Ctrl+Shift+W")
        close_all.triggered.connect(self.display_window_manager_class.close_all)
        window_menu.addAction(close_all)

    def _create_bottom_bar(self):
        widget = QWidget()
        layout = QHBoxLayout()
        btn_load = QPushButton("Load any config")
        btn_load.clicked.connect(self._load_any_config)
        btn_save = QPushButton("Save config")
        btn_save.clicked.connect(self._save_config)
        btn_open = QPushButton("Open reconstruction")
        btn_open.clicked.connect(self._open_reconstruction)
        btn_run = QPushButton("Run")
        btn_run.clicked.connect(self._run)
        layout.addWidget(btn_load)
        layout.addWidget(btn_save)
        layout.addStretch()
        layout.addWidget(btn_open)
        layout.addWidget(btn_run)
        widget.setLayout(layout)
        return widget

    # ── shared actions ──────────────────────────────────────────────────

    ## refreshes every section's UI from the cached config file
    def load_cached_config(self):
        for section in self._all_sections:
            if hasattr(section, 'update_ui_from_toml'):
                section.update_ui_from_toml(self.cached_config_path)

    def _load_any_config(self):
        path = open_file(self, "Select config", self.results_dir, "*.toml")
        if path:
            config = self.load_config(path)
            self.save_config(config, self.cached_config_path)
            self.load_cached_config()

    def _save_config(self):
        path = save_file(self, "Save config", self.results_dir, "*.toml")
        if path:
            config = self.load_config(self.cached_config_path)
            self.save_config(config, path)

    def _run(self):
        config = self.load_config(self.cached_config_path)
        pipeline = self.pipeline_class.create(config)
        display_window = self.display_window_manager_class.create(pipeline)
        display_window.show()
        pipeline.start()

    def _open_reconstruction(self):
        from os.path import join
        open_dir = open_directory(self, "Select Folder", self.results_dir)
        if not open_dir:
            return
        config = self.load_config(join(open_dir, 'config.toml'))
        pipeline = self.pipeline_class.create(config)
        pipeline.load_results(open_dir)
        display_window = self.display_window_manager_class.create(pipeline)
        display_window.show()
        display_window.initialize_from_existing_data()

    # ── close / key events ──────────────────────────────────────────────

    def on_close_cleanup(self):
        pass

    def closeEvent(self, event):
        self.on_close_cleanup()
        self.display_window_manager_class.close_all()
        self.pipeline_class.stop_all()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            widget = self.focusWidget()
            if widget:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)
