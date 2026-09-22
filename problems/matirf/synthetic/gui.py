"""
The ground-truth generator window — `matirf synth`.

    left    one section per part of the scene (cache/scene.toml): the grid, the seed, then
            one per kind of object, whose parameters appear only when its count is > 0
    right   the preview (the MA-TIRF display window's figures: depth map + profiles, or
            image + histogram), and the actions:

                Generate / preview          draw the scene again from the current settings
                Save truth…                 the TIF and its <name>.truth.json record
                Use as MA-TIRF truth        the same, then points the MA-TIRF config at it
                                            (synthetic mode, nz / z0 / zN from the grid, and
                                            synthetic/measurement_parameters.json as the
                                            microscope that simulates g)
                Load scene… / Save scene…   a scene TOML — presets/ holds ready-made ones

Every edit is written to the scene file at once, as in the control window.
"""

import sys

from PyQt5.QtWidgets import (
    QApplication, QStyleFactory, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QMessageBox,
)

import settings as settings
from fileio import load_or_create_toml
from gui.base.base_section_qgroup import BaseSectionQGroup
from gui.errors import install_error_handlers
from gui.file_dialog import open_file, save_file
from gui.factory import figures_section_class
from problems.matirf import MATIRF_SYNTHETIC_DIR
from problems.matirf.cache import update_cache as update_matirf_cache
from .grid import GRID_UI, GRID_TOML_KEY
from .objects import OBJECTS, gate_by_count
from .scene import (
    SCENE_PATH, PRESETS_DIR, DEFAULT_SCENE, SAMPLING_UI, SAMPLING_TOML_KEY,
    load_scene, save_scene, update_scene_cache,
)
from .generator import generate, save_truth


def _figures_section():
    """The MA-TIRF figures section — imported late: the matirf package is still loading."""
    from problems.matirf import MATIRF
    return figures_section_class(MATIRF)


class SyntheticTruthGeneratorWindow(QWidget):
    """Edit a scene, preview its ground truth, save it for MA-TIRF."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MA-TIRF — Synthetic Ground Truth Generator")
        self.f_true, self.figures, self._last_shape = None, None, None
        load_scene()                                  # creates / completes the cache file
        self._sections = []
        self._build_ui()
        self._generate()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        column = QVBoxLayout()
        column.addWidget(self._section("Grid", GRID_UI, GRID_TOML_KEY))
        column.addWidget(self._section("Sampling", SAMPLING_UI, SAMPLING_TOML_KEY))
        for key, cls in OBJECTS.items():
            column.addWidget(self._section(cls.name, gate_by_count(cls.ui_params), key))
        column.addStretch()
        panel = QWidget()
        panel.setLayout(column)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFixedWidth(430)
        root.addWidget(scroll)

        right = QVBoxLayout()
        self.viewer_box = QVBoxLayout()
        viewer = QWidget()
        viewer.setLayout(self.viewer_box)
        right.addWidget(viewer, stretch=1)
        right.addLayout(self._actions())
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        root.addLayout(right, stretch=1)
        self.resize(1250, 880)

    def _section(self, title, ui, key):
        section = BaseSectionQGroup(
            parent=self, update_cache_fn=update_scene_cache,
            load_toml_fn=lambda path: load_or_create_toml(path, DEFAULT_SCENE),
            title=title, params_ui_dict=ui, toml_section_key=key, config_path=SCENE_PATH)
        section.update_ui_from_toml(SCENE_PATH)
        self._sections.append(section)
        return section

    def _actions(self):
        buttons = QHBoxLayout()
        self.btn_generate = self._button(buttons, "Generate / preview", self._generate)
        self.btn_save = self._button(buttons, "Save truth…", self._save_truth)
        self.btn_use = self._button(buttons, "Use as MA-TIRF truth", self._use_as_truth)
        self.btn_load = self._button(buttons, "Load scene…", self._load_scene)
        self.btn_save_scene = self._button(buttons, "Save scene…", self._save_scene)
        return buttons

    @staticmethod
    def _button(layout, label, action):
        button = QPushButton(label)
        button.clicked.connect(action)
        layout.addWidget(button)
        return button

    # ── actions ──────────────────────────────────────────────────────────────

    def _generate(self):
        self.status.setText("Generating…")
        QApplication.processEvents()
        scene = load_scene()
        grid = scene[GRID_TOML_KEY]
        if grid["zN_nm"] <= grid["z0_nm"]:
            self.status.setText("zN must be greater than z0.")
            return
        self.f_true = generate(scene)
        self._show(self.f_true, grid)
        self.status.setText(
            f"Truth {tuple(self.f_true.shape)} (Z, Y, X) in [0, 1], seed "
            f"{scene[SAMPLING_TOML_KEY]['seed']} — reconstructions will use {grid['nz']} planes "
            f"({grid['z_oversampling']} truth planes each), between {grid['z0_nm']:g} and "
            f"{grid['zN_nm']:g} nm.")

    def _show(self, f, grid):
        """Rebuild the figures only when the shape changes; otherwise refresh them in place."""
        config = {"oper-params": {"z0": grid["z0_nm"], "zN": grid["zN_nm"]}}
        if self.figures is None or tuple(f.shape) != self._last_shape:
            while self.viewer_box.count():
                widget = self.viewer_box.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()
            self.figures = _figures_section()(parent=self)
            self.viewer_box.addWidget(self.figures)
            self._last_shape = tuple(f.shape)
        self.figures.update_plot(f, config)

    def _save(self, default_path, title):
        if self.f_true is None:
            return None
        path = save_file(self, title, default_path, "TIFF (*.TIF *.tif)")
        return save_truth(self.f_true, path, load_scene()) if path else None

    def _save_truth(self):
        path = self._save(str(MATIRF_SYNTHETIC_DIR), "Save the synthetic truth")
        if path:
            self.status.setText(f"Saved {path} and its record {path.with_suffix('.truth.json').name}")

    def _use_as_truth(self):
        path = self._save(str(MATIRF_SYNTHETIC_DIR / "synthetic_truth.TIF"),
                          "Save the synthetic truth for MA-TIRF")
        if not path:
            return
        grid = load_scene()[GRID_TOML_KEY]
        update_matirf_cache(["input-paths", "mode"], "synthetic-data")
        update_matirf_cache(["input-paths", "tif"], str(path))
        update_matirf_cache(["input-paths", "json"], str(MATIRF_SYNTHETIC_DIR / "measurement_parameters.json"))
        update_matirf_cache(["oper-params", "nz"], int(grid["nz"]))
        update_matirf_cache(["oper-params", "z0"], float(grid["z0_nm"]))
        update_matirf_cache(["oper-params", "zN"], float(grid["zN_nm"]))
        QMessageBox.information(
            self, "Synthetic truth set",
            f"The MA-TIRF config now uses this truth in synthetic mode, reconstructed on "
            f"{grid['nz']} planes between {grid['z0_nm']:g} and {grid['zN_nm']:g} nm.\n"
            f"It is simulated with the microscope of synthetic/measurement_parameters.json.\n"
            f"Still to choose: the noise and an algorithm.")
        self.status.setText(f"Set as the MA-TIRF synthetic truth: {path}")

    def _load_scene(self):
        path = open_file(self, "Load a scene", str(PRESETS_DIR), "*.toml")
        if not path:
            return
        save_scene(load_scene(path), SCENE_PATH)
        for section in self._sections:
            section.update_ui_from_toml(SCENE_PATH)
        self._generate()

    def _save_scene(self):
        path = save_file(self, "Save the scene", str(PRESETS_DIR), "*.toml")
        if path:
            path = path if path.lower().endswith(".toml") else path + ".toml"
            save_scene(load_scene(), path)
            self.status.setText(f"Scene saved to {path}")


def main():
    app = QApplication(sys.argv)
    install_error_handlers()                  # an error prints its traceback, never aborts
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette
    app.setPalette(palette())
    window = SyntheticTruthGeneratorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
