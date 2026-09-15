"""
Test and reference implementation of the ``common/gui/specializable`` layer.

Run it directly:
    python -m common.gui.specializable._tests
    (or:  python common/gui/specializable/_tests.py)


Unlike ``base/`` and ``reusable/`` (whose objects are concrete and usable as-is), every
class in ``specializable/`` is an ABSTRACT skeleton with ``raise NotImplementedError``
hooks — it is meant to be subclassed by a concrete inverse problem (this is exactly what
``matirf/`` and ``deconv/`` do). So it cannot be instantiated directly.

The honest way to test the skeleton is therefore to provide the SMALLEST possible
specialization that fills the hooks. That is what this file is: a self-contained "dummy
inverse problem" that wires up every Base* class. It doubles as living documentation of
the extension contract described in ``common/gui/specializable/__init__.py``:

    BaseControlWindow          -> DummyControlWindow        (declarative: only class attrs)
    BaseInputFilesSection      -> DummyInputFilesSection    (8 hooks stubbed)
    BaseDisplayWindow          -> DummyDisplayWindow        (figures/pipeline hooks)
    BaseFiguresSection         -> FakeFiguresSection        (create_views / update_views)
    BaseDisplayWindowManager   -> FakeManager               (_create_window)
    + a FakePipeline standing in for the whole backend (no real algorithm, no real data).


Two windows:
    > the CONTROL window, fully assembled from section descriptors (a demo parameter
      section, the reused AddNoiseSection, the reused AlgorithmSelectionSection with a
      fake algorithm registry, and the DummyInputFilesSection). Click "Run" to see the
      whole pipeline -> Qt bridge -> display-window flow fire with fakes.
    > a DISPLAY window, pre-populated via the fake pipeline, showing the figures section
      (multi-view switch), the config/messages panels, and the save button.

Everything is backed by a throwaway TOML (path printed on start). Nothing touches the real app.
"""

import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QStyleFactory, QWidget, QLabel, QVBoxLayout,
)

import common.settings as settings
from common import DataMode
from common.core.enums import PipelineState

from common.gui.reusable import AddNoiseSection, AlgorithmSelectionSection
from common.gui.specializable.control_window import BaseControlWindow, BaseInputFilesSection
from common.gui.specializable.display_window import (
    BaseDisplayWindow, BaseDisplayWindowManager, BaseFiguresSection,
)


# ── throwaway config location (never touches the real app caches) ─────────────
TMP_DIR = Path(tempfile.mkdtemp())
TMP_TOML = TMP_DIR / "demo_cache.toml"


# ── demo parameter section + a fake algorithm registry (test data) ────────────
DEMO_SECTION_UI = {
    "count": {"title": "How many objects", "type": "value",
              "param_info": {'dtype': int, 'unit': '', 'latex_name': 'n', 'default': 2}},
    "size": {"title": "Object size (shown only if count > 0)", "type": "value",
             "depends_on": {"count": lambda v: (v or 0) > 0},
             "param_info": {'dtype': float, 'unit': 'nm', 'latex_name': 's', 'default': 100.0}},
    "kind": {"title": "Kind", "type": "option", "param_info": {'options_list': ['A', 'B', 'C']}},
}


class _FakeAlgo:
    _ui = {}
    @classmethod
    def get_ui_params(cls, features=None):
        return cls._ui

class _AlgoAlpha(_FakeAlgo):
    _ui = {"iter": {"title": "Iterations", "type": "value",
                    "param_info": {'dtype': int, 'unit': '', 'latex_name': 'n', 'default': 20}}}

class _AlgoBeta(_FakeAlgo):
    _ui = {"sigma": {"title": "Sigma", "type": "value",
                     "param_info": {'dtype': float, 'unit': '', 'latex_name': '\\sigma', 'default': 15.0}}}

FAKE_ALGORITHMS = {"Alpha": _AlgoAlpha, "Beta": _AlgoBeta}


# a TOML-clean default config covering every section the control window builds:
DUMMY_DEFAULT_CONFIG = {
    'algorithm': 'None',
    'algo-params': {},
    'demo-section': {'count': 2, 'size': 100.0, 'kind': 'A'},
    'add-noise': {'add_noise': False, 'is_gaussian': True, 'sigma': 'null'},
    'input-paths': {'mode': DataMode.REAL.value},
}


# ── the fake backend: a pipeline + result standing in for the whole algorithm stack ──
class FakeResult:
    def __init__(self):
        self.f = "dummy-reconstruction"       # any object: FakeFiguresSection just displays it
        self.messages = "[dummy] restored messages from a fake run"
        self.diff = None
        self.metrics = None
    def has_synthetic_truth(self):
        return False


class FakePipeline:
    """Minimal stand-in for a BasePipeline: only what BaseDisplayWindow / BaseControlWindow read."""
    _pipelines = []

    def __init__(self, config):
        self.config = config
        self.is_synthetic_data = False
        self.algorithm = None                 # BaseDisplayWindow polls .algorithm._latest_f
        self.result = FakeResult()
        # callbacks are (re)wired by BaseDisplayWindow._connect_pipeline_callbacks:
        self.on_message = lambda msg: None
        self.on_finished = lambda result: None
        self.on_error = lambda err: None
        self.on_state_changed = lambda old, new: None

    @classmethod
    def create(cls, config):
        p = cls(config)
        cls._pipelines.append(p)
        return p

    @classmethod
    def stop_all(cls):
        cls._pipelines.clear()

    @classmethod
    def remove(cls, pipeline):
        if pipeline in cls._pipelines:
            cls._pipelines.remove(pipeline)

    def start(self):
        # emulate a run so the pipeline -> Qt bridge -> UI wiring is visibly exercised:
        self.on_state_changed(PipelineState.IDLE, PipelineState.COMPUTING)
        self.on_message("[dummy] pipeline started")
        self.on_message("[dummy] iter 1 ... iter 2 ... done")
        self.on_finished(self.result)
        self.on_state_changed(PipelineState.COMPUTING, PipelineState.COMPLETED)

    def stop(self):
        pass

    def save_results(self, save_dir):
        self.on_message(f"[dummy] would save results to {save_dir}")


# ── the minimal specializations of each Base* skeleton ────────────────────────
class FakeFiguresSection(BaseFiguresSection):
    """Two placeholder views, to show the multi-view switch header the base class adds."""

    def create_views(self):
        f = getattr(self, "_last_f", None)
        self._v1 = QLabel(f"View 1 — reconstruction\n(f = {f})")
        self._v2 = QLabel("View 2 — another representation")
        for v in (self._v1, self._v2):
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumSize(320, 240)
            v.setStyleSheet("border: 1px dashed gray;")
        return [self._v1, self._v2]

    def update_views(self, f, config):
        # called before create_views on the first update -> just remember the data:
        self._last_f = f
        if hasattr(self, "_v1"):
            self._v1.setText(f"View 1 — reconstruction\n(f = {f})")

    def view_labels(self):
        return ["view 1", "view 2"]


class FakeManager(BaseDisplayWindowManager):
    @classmethod
    def _create_window(cls, pipeline):
        return DummyDisplayWindow(pipeline)


class DummyDisplayWindow(BaseDisplayWindow):
    def window_title(self):
        return "Dummy DISPLAY window (specializable demo)"
    def results_dir(self):
        return str(TMP_DIR)
    def pipeline_class(self):
        return FakePipeline
    def display_window_manager_class(self):
        return FakeManager
    def create_figures_section(self):
        return FakeFiguresSection(parent=self)
    def update_figures(self, f, config):
        self.figures_section.update_plot(f, config)


class _DummySelector(QWidget):
    """Stand-in for a real file selector: the interface BaseInputFilesSection expects."""
    def __init__(self, label):
        super().__init__()
        self.is_file_selected = False
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"[{label} selector]\n(dummy)"))
    def update_mode(self):
        pass
    def update_selected_file(self, path):
        self.is_file_selected = bool(path and path != "None")


class DummyInputFilesSection(BaseInputFilesSection):
    def _get_cached_mode(self):
        return True                                   # start in "real" mode
    def _real_mode_text(self):
        return "Real data"
    def _synthetic_mode_text(self):
        return "Synthetic"
    def _create_image_selector(self):
        return _DummySelector("image")
    def _create_json_selector(self):
        return _DummySelector("JSON params")
    def _on_switch_mode(self, is_mode_real):
        print(f"[dummy] input-files mode switched -> {'real' if is_mode_real else 'synthetic'}")
    def _load_config_for_update(self, toml_path):
        from common.in_out import load_or_create_toml
        return load_or_create_toml(toml_path, DUMMY_DEFAULT_CONFIG)
    def _get_file_paths_from_config(self, config):
        return (None, None)


class DummyControlWindow(BaseControlWindow):
    """The whole point: a control window configured ENTIRELY through class attributes."""

    window_title = "Dummy CONTROL window (specializable demo) — click Run"
    cached_config_path = TMP_TOML
    default_config = DUMMY_DEFAULT_CONFIG
    results_dir = str(TMP_DIR)
    pipeline_class = FakePipeline
    display_window_manager_class = FakeManager

    # Same section ordering as the real matirf / deconv control windows:
    #   left column  -> input files FIRST, then the parameter sections (add-noise, ...)
    #   right column -> the algorithm selection
    sections_left = [
        DummyInputFilesSection,                              # input files at the top (like matirf/deconv)
        ("Demo section", DEMO_SECTION_UI, "demo-section"),   # (title, ui_dict, toml_key)
        AddNoiseSection,                                     # reused from reusable/
    ]
    sections_right = [
        (AlgorithmSelectionSection,                          # reused, with extra kwargs
         {'algorithms_dict': FAKE_ALGORITHMS, 'problem_features': None}),
    ]


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    print("common/gui/specializable walkthrough — demo cache TOML:", TMP_TOML)

    # 1) the CONTROL window: assembled purely from class attributes / section descriptors.
    control = DummyControlWindow()
    control.show()

    # 2) a DISPLAY window, pre-populated through the fake pipeline + manager, so the
    #    display skeleton is exercised without having to click "Run":
    pipeline = FakePipeline.create(DUMMY_DEFAULT_CONFIG)
    display = FakeManager.create(pipeline)
    display.show()
    display.initialize_from_existing_data()   # fills figures + messages, enables Save

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
