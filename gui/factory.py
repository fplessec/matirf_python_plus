"""
Building both windows of an inverse problem from its declaration.

This is the module that removes the nine interface files each problem used to carry. Every
one of them was a subclass whose whole content was a handful of class attributes; here those
attributes are read from `problem.ui` and the subclasses are created at import time instead
of being typed out.

    control_window_class(problem)   -> a ready ControlWindow class
    display_window_class(problem)   -> a ready DisplayWindow class
    window_manager_class(problem)   -> the manager that owns the open display windows

The generic base classes are unchanged: `BaseControlWindow`, `BaseDisplayWindow`,
`BaseInputFilesSection`, `BaseFiguresSection`, `BaseSyntheticTruthSection` already took
their configuration from class attributes. What was missing was someone to fill them in.
Nothing about how a window is DRAWN lives here — only which declaration feeds which
attribute.

Each class is built once per problem and cached. That is not an optimization: the control
window and the display window both ask for the problem's pipeline class and its manager,
and handing them different classes would give each its own registry of running
reconstructions — a window would then try to remove a pipeline from a registry it was never
added to, and the run would linger forever.
"""

from gui.reusable import AddNoiseSection, AlgorithmSelectionSection  # noqa: F401
from gui.specializable.control_window import BaseControlWindow, BaseInputFilesSection
from gui.specializable.display_window import (
    BaseDisplayWindow, BaseDisplayWindowManager, BaseFiguresSection,
)
from gui.specializable.display_window.sections import BaseSyntheticTruthSection
from gui.estimators import attach


_CACHE = {}


def _cached(kind, problem, build):
    key = (kind, problem.name)
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── the sections ──────────────────────────────────────────────────────────────

def input_files_section_class(problem) -> type:
    """The input-files section: the mode toggle and the two file selectors."""
    def build():
        from fileio.cache import make_update_cache
        ui = problem.ui
        ## The class name is deliberately the SAME for every problem, because
        ## BaseControlWindow derives the attribute name from it: the section is always
        ## `window.input_files_section`, never `window.matirf_input_files_section`. A name
        ## that changed with the problem would make generic code — and every test — have to
        ## know which problem it is looking at.
        return type(
            "InputFilesSection",
            (BaseInputFilesSection,),
            {
                "CONFIG_PATH": problem.config_path,
                "DEFAULT_CONFIG": problem.default_config,
                "UPDATE_CACHE_FN": staticmethod(
                    make_update_cache(problem.config_path, problem.default_config)),
                "MEASUREMENTS_DIR": problem.measurements_dir,
                "REAL_MODE_TEXT": ui.real_mode_text,
                "SYNTHETIC_MODE_TEXT": ui.synthetic_mode_text,
                ## which side of the toggle a fresh config starts on:
                "DEFAULT_MODE_REAL":
                    (problem.default_config or {}).get("input-paths", {})
                    .get("mode", "real-data") == "real-data",
                "IMAGE_SLOT": ui.image_slot,
                "JSON_SLOT": ui.json_slot,
                "__doc__": f"Input files of the {problem.name!r} problem (generated from its ui).",
            },
        )
    return _cached("inputs", problem, build)


def figures_section_class(problem) -> type:
    """The switchable views of the display window."""
    def build():
        ui = problem.ui
        return type(
            f"{problem.name.capitalize()}FiguresSection",
            (BaseFiguresSection,),
            {
                "VIEWS": list(ui.views),
                "EXPORT_DEFAULT_DIR": ui.export_default_dir or problem.results_dir,
                "__doc__": f"Figures of the {problem.name!r} problem (generated from its ui).",
            },
        )
    return _cached("figures", problem, build)


def synthetic_section_class(problem):
    """The metrics table and the difference popup — None when the problem has no truth."""
    ui = problem.ui
    if not ui.supports_synthetic_view:
        return None

    def build():
        return type(
            f"{problem.name.capitalize()}SyntheticTruthSection",
            (BaseSyntheticTruthSection,),
            {"DIFFERENCE_VIEWER_CLASS": ui.difference_viewer,
             "__doc__": f"Synthetic-truth analysis of the {problem.name!r} problem."},
        )
    return _cached("synthetic", problem, build)


# ── the windows ───────────────────────────────────────────────────────────────

def window_manager_class(problem) -> type:
    """Owns the display windows currently open for this problem."""
    def build():
        def _create_window(cls, pipeline):
            return display_window_class(problem)(pipeline)
        return type(
            f"{problem.name.capitalize()}DisplayWindowManager",
            (BaseDisplayWindowManager,),
            {"_create_window": classmethod(_create_window),
             "__doc__": f"Display windows open for the {problem.name!r} problem."},
        )
    return _cached("manager", problem, build)


def display_window_class(problem) -> type:
    """The window that shows a running or finished reconstruction."""
    def build():
        from pipeline import pipeline_for
        ui = problem.ui
        return type(
            f"{problem.name.capitalize()}DisplayWindow",
            (BaseDisplayWindow,),
            {
                "WINDOW_TITLE": ui.display_title,
                "RESULTS_DIR": problem.results_dir,
                "PIPELINE_CLASS": pipeline_for(problem),
                "DISPLAY_WINDOW_MANAGER_CLASS": window_manager_class(problem),
                "FIGURES_SECTION_CLASS": figures_section_class(problem),
                "SYNTHETIC_SECTION_CLASS": synthetic_section_class(problem),
                "__doc__": f"Display window of the {problem.name!r} problem.",
            },
        )
    return _cached("display", problem, build)


def control_window_class(problem) -> type:
    """The window where a run is configured and started."""
    def build():
        from pipeline import pipeline_for
        from solvers import SOLVERS
        ui = problem.ui
        inputs = input_files_section_class(problem)

        ## the solver registry, seen through this problem's Estimate buttons:
        solvers_view = attach(SOLVERS, ui.estimators, problem)

        ## Input files first, then the problem's own parameter sections in declaration
        ## order, then the noise section — which is NOT declared by a problem because it
        ## belongs to the framework: simulating a measurement and corrupting it with a
        ## chosen noise model is what `[add-noise]` means for every inverse problem.
        ## A problem opts out simply by having no '[add-noise]' section in its default config.
        sections_left = [inputs] + list(ui.parameters)
        if "add-noise" in (problem.default_config or {}):
            from gui.reusable import ADD_NOISE_PARAMETERS_UI
            sections_left.append(
                ("Add noise to measurement", ADD_NOISE_PARAMETERS_UI, "add-noise"))
        sections_right = [(AlgorithmSelectionSection,
                           {"algorithms_dict": solvers_view,
                            "problem_features": problem.features})]

        def on_close_cleanup(self):
            """Close the preview / editor popups the input selectors may have opened."""
            section = getattr(self, _attribute_name(inputs), None)
            if section is None:
                return
            section.json_selector.close_sub_window()
            section.image_selector.close_sub_window()

        return type(
            f"{problem.name.capitalize()}ControlWindow",
            (BaseControlWindow,),
            {
                "window_title": ui.control_title,
                "cached_config_path": problem.config_path,
                "default_config": problem.default_config,
                "results_dir": problem.results_dir,
                "pipeline_class": pipeline_for(problem),
                "display_window_manager_class": window_manager_class(problem),
                "sections_left": sections_left,
                "sections_right": sections_right,
                "on_close_cleanup": on_close_cleanup,
                "__doc__": f"Control window of the {problem.name!r} problem "
                           f"(generated from its ui declaration).",
            },
        )
    return _cached("control", problem, build)


def _attribute_name(section_class) -> str:
    """The attribute BaseControlWindow assigns a section class to (CamelCase -> snake_case)."""
    from gui.specializable.control_window.base_control_window import _camel_to_snake
    return _camel_to_snake(section_class.__name__)
