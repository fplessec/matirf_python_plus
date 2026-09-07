"""
Synthetic Ground Truth Generator window.

Built exactly like the control window: a column of QGroupBox sections, each a
BaseSectionQGroup rendering a params_ui_dict of SimpleParameterWidgets bound to a TOML.

    Left column, driven by two TOML files:
        > Grid          (grid.toml)          — how the truth is sampled / visualized
        > Sampling      (ground_truth.toml)  — the master seed
        > Ellipsoid / Filament / Membrane / Double layer (ground_truth.toml)
                        — one section per object type: 'count' + characteristic params

    Right column:
        > preview via the MA-TIRF display window's FiguresSection (two switchable views:
          DepthMap+Profiles / Image+Histogram 3D)
        > actions: generate, save TIF, use as MA-TIRF synthetic truth, load/save configs

Launch with 'matirf synth' or 'python -m matirf.synthetic'.
"""

import sys
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QStyleFactory, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QMessageBox,
)

import common.settings as settings
from common.in_out import load_or_create_toml, save_toml, save_tif
from common.gui.base.base_section_qgroup import BaseSectionQGroup
from common.gui.file_dialog import open_file, save_file
from matirf import MATIRF_MEASUREMENTS_DIR, MATIRF_RESULTS_DIR
from matirf.cache import update_cache as update_matirf_cache
from matirf.gui.display_window.sections.figures_section import FiguresSection
from .objects import OBJECT_TYPES
from .objects.base import gate_by_count
from .grid import GRID_UI, GRID_TOML_KEY
from .config import (
    GRID_CONFIG_PATH, GT_CONFIG_PATH, DEFAULT_GRID_CONFIG, DEFAULT_GT_CONFIG,
    SAMPLING_UI, SAMPLING_TOML_KEY, load_grid_config, load_gt_config,
    update_grid_cache, update_gt_cache,
)
from .generator import generate_ground_truth


class SyntheticTruthGeneratorWindow(QWidget):
    """Interactive, TOML-driven designer for a reproducible MA-TIRF synthetic ground truth."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MA-TIRF — Synthetic Ground Truth Generator")
        self.f_true = None
        self.figures = None
        self._last_shape = None
        # ensure both cache files exist with defaults before wiring widgets
        load_grid_config()
        load_gt_config()
        # load functions (path -> dict) bound to each file's defaults
        self._load_grid = partial(load_or_create_toml, default_config=DEFAULT_GRID_CONFIG)
        self._load_gt = partial(load_or_create_toml, default_config=DEFAULT_GT_CONFIG)
        self._sections = []   # list of (section, config_path) to refresh on load
        self._build_ui()
        self._generate()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        # left: scrollable column of parameter sections
        panel = QVBoxLayout()
        panel.addWidget(self._section("Grid (visualization)", GRID_UI, GRID_TOML_KEY,
                                      update_grid_cache, self._load_grid, GRID_CONFIG_PATH))
        panel.addWidget(self._section("Sampling", SAMPLING_UI, SAMPLING_TOML_KEY,
                                      update_gt_cache, self._load_gt, GT_CONFIG_PATH))
        for key, cls in OBJECT_TYPES.items():
            # gate_by_count -> the whole parameter set is shown only when count > 0
            section = self._section(cls.name, gate_by_count(cls.ui_params), cls.toml_key,
                                    update_gt_cache, self._load_gt, GT_CONFIG_PATH)
            self._wire_count_prune(section, cls.toml_key)
            panel.addWidget(section)
        panel.addStretch()
        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel_widget)
        scroll.setFixedWidth(400)
        root.addWidget(scroll)

        # right: preview + actions
        right = QVBoxLayout()
        self.viewer_box = QVBoxLayout()
        viewer_container = QWidget()
        viewer_container.setLayout(self.viewer_box)
        right.addWidget(viewer_container, stretch=1)
        right.addLayout(self._actions())
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        root.addLayout(right, stretch=1)

        self.resize(1200, 850)

    def _section(self, title, ui_dict, toml_key, update_fn, load_fn, config_path):
        section = BaseSectionQGroup(
            parent=self, update_cache_fn=update_fn, load_toml_fn=load_fn,
            title=title, params_ui_dict=ui_dict, toml_section_key=toml_key,
            config_path=config_path,
        )
        section.update_ui_from_toml(config_path)
        self._sections.append((section, config_path))
        return section

    def _wire_count_prune(self, section, toml_key):
        """When an object's count drops to 0, collapse its TOML section to just {count}."""
        count_widget = section.parameter_widgets.get("count")
        if count_widget is None or count_widget.input_widget is None:
            return
        count_widget.input_widget.textChanged.connect(
            lambda _=None, k=toml_key: self._prune_if_zero(k))

    def _prune_if_zero(self, toml_key):
        config = load_or_create_toml(GT_CONFIG_PATH, DEFAULT_GT_CONFIG)
        section = config.get(toml_key, {})
        try:
            count = int(section.get("count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count <= 0 and set(section.keys()) != {"count"}:
            config[toml_key] = {"count": count}
            save_toml(config, GT_CONFIG_PATH)

    def _actions(self):
        col = QVBoxLayout()
        row1 = QHBoxLayout()
        self.btn_generate = QPushButton("Generate / preview")
        self.btn_generate.clicked.connect(self._generate)
        self.btn_save = QPushButton("Save as TIF…")
        self.btn_save.clicked.connect(self._save_tif)
        self.btn_use = QPushButton("Use as MA-TIRF synthetic truth")
        self.btn_use.clicked.connect(self._use_as_truth)
        for b in (self.btn_generate, self.btn_save, self.btn_use):
            row1.addWidget(b)
        row2 = QHBoxLayout()
        for label, path, default in (
            ("Load ground-truth .toml", GT_CONFIG_PATH, DEFAULT_GT_CONFIG),
            ("Save ground-truth .toml", GT_CONFIG_PATH, DEFAULT_GT_CONFIG),
            ("Load grid .toml", GRID_CONFIG_PATH, DEFAULT_GRID_CONFIG),
            ("Save grid .toml", GRID_CONFIG_PATH, DEFAULT_GRID_CONFIG),
        ):
            btn = QPushButton(label)
            if label.startswith("Load"):
                btn.clicked.connect(partial(self._load_config, path, default))
            else:
                btn.clicked.connect(partial(self._save_config, path, default))
            row2.addWidget(btn)
        col.addLayout(row1)
        col.addLayout(row2)
        return col

    # ── logic ───────────────────────────────────────────────────────────

    def _generate(self):
        self.status.setText("Generating…")
        QApplication.processEvents()
        grid_cfg = load_grid_config()
        gt_cfg = load_gt_config()
        if grid_cfg[GRID_TOML_KEY]["zN_nm"] <= grid_cfg[GRID_TOML_KEY]["z0_nm"]:
            self.status.setText("⚠ zN must be greater than z0.")
            return
        self.f_true = generate_ground_truth(grid_cfg, gt_cfg)
        self._show_preview(self.f_true, grid_cfg[GRID_TOML_KEY])
        seed = gt_cfg.get(SAMPLING_TOML_KEY, {}).get("seed", 0)
        self.status.setText(
            f"Generated {tuple(self.f_true.shape)} (Z,Y,X), range [0, 1]. Seed {seed}."
        )

    def _show_preview(self, f, grid_params):
        z0, zN = grid_params["z0_nm"], grid_params["zN_nm"]
        config = {'oper-params': {'z0': z0, 'zN': zN}}
        shape = tuple(f.shape)
        if self.figures is None or shape != self._last_shape:
            while self.viewer_box.count():
                item = self.viewer_box.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self.figures = FiguresSection(parent=self)
            self.viewer_box.addWidget(self.figures)
            self.figures.update_plot(f, config)
            self._last_shape = shape
        else:
            for w in (self.figures.depth_map_widget, self.figures.profiles_widget):
                w.z0, w.zN = z0, zN
                w.set_image(f)
            self.figures.viewer_3d.set_image(f)

    def _refresh_sections(self):
        for section, config_path in self._sections:
            section.update_ui_from_toml(config_path)

    def _save_tif_to(self, default_path, title):
        path = save_file(self, title, default_path, "TIFF (*.TIF *.tif)")
        if not path:
            return None
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".TIF"
        save_tif(self.f_true, path)
        return path

    def _save_tif(self):
        if self.f_true is None:
            return
        path = self._save_tif_to(str(MATIRF_MEASUREMENTS_DIR), "Save synthetic truth")
        if path:
            self.status.setText(f"Saved to {path}")

    def _use_as_truth(self):
        if self.f_true is None:
            return
        default = str(MATIRF_MEASUREMENTS_DIR / "synthetic_truth.TIF")
        path = self._save_tif_to(default, "Save synthetic truth for MA-TIRF")
        if not path:
            return
        grid = load_grid_config()[GRID_TOML_KEY]
        update_matirf_cache(['input-paths', 'mode'], "synthetic-data")
        update_matirf_cache(['input-paths', 'tif'], path)
        update_matirf_cache(['oper-params', 'nz'], int(grid["nz"]))
        update_matirf_cache(['oper-params', 'z0'], float(grid["z0_nm"]))
        update_matirf_cache(['oper-params', 'zN'], float(grid["zN_nm"]))
        QMessageBox.information(
            self, "Synthetic truth set",
            "The MA-TIRF config now uses this ground truth in synthetic mode.\n"
            "You still need a measurement JSON and an algorithm; noise is added by the "
            "reconstruction pipeline via the [add-noise] section.",
        )
        self.status.setText(f"Set as synthetic truth: {path}")

    def _load_config(self, cache_path, default_config):
        path = open_file(self, "Select .toml config", MATIRF_RESULTS_DIR, "*.toml")
        if not path:
            return
        config = load_or_create_toml(path, default_config)
        save_toml(config, cache_path)
        self._refresh_sections()
        self._generate()

    def _save_config(self, cache_path, default_config):
        path = save_file(self, "Save .toml config", MATIRF_RESULTS_DIR, "*.toml")
        if not path:
            return
        if not path.lower().endswith(".toml"):
            path += ".toml"
        save_toml(load_or_create_toml(cache_path, default_config), path)
        self.status.setText(f"Saved config to {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette
    app.setPalette(palette())
    window = SyntheticTruthGeneratorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
