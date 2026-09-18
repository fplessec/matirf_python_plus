"""
Test of the ``common/gui/base`` layer.

Run it directly:
    python -m gui.base._tests
    (or:  python common/gui/base/_tests.py)

This single window is meant for someone discovering the code. It puts the three levels
of the parameter-UI hierarchy side by side, in the exact order described in
``common/gui/base/__init__.py``, so the composition is visible at a glance:

    1. SimpleParameterWidget   one parameter dict   ->  one line          (the atom)
    2. BaseSectionWidget       a full UI dict       ->  stacked lines     (composition)
    3. BaseSectionQGroup       the same UI dict     ->  lines in a titled, bordered frame
                                                                          (visual grouping)

Reading the three columns left-to-right shows what each layer adds:
    > level 1 renders/holds ONE value (here shown with no TOML backing: an atom is usable
      on its own; persistence is optional);
    > level 2 builds a whole section from a dictionary, stacks the atoms with separators,
      and wires the ``depends_on`` show/hide logic (edit ``count`` to 0 -> ``size`` hides);
    > level 3 is the very same section, wrapped in a QGroupBox — the titled border is ALL
      it adds on top of level 2.

Levels 2 and 3 are backed by the SAME throwaway TOML (path printed to the console on start):
editing any value writes to it, so you can open it to watch the config-sync in action.
Nothing here touches the real application caches.
"""

import sys
import tempfile
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QStyleFactory, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
)

import settings as settings
from fileio.cache import make_update_cache
from fileio import load_or_create_toml

from gui.base.single_parameter_widget import SimpleParameterWidget
from gui.base.base_section_widget import BaseSectionWidget
from gui.base.base_section_qgroup import BaseSectionQGroup


# ── the demo section (test data, deliberately kept here and not in the library files) ──
# It exercises every parameter type AND a callable 'depends_on' ('size' is shown only
# when 'count' > 0).
DEMO_SECTION_UI = {
    "count": {"title": "How many objects", "type": "value",
              "param_info": {'dtype': int, 'unit': '', 'latex_name': 'n', 'default': 2}},
    "size": {"title": "Object size (shown only if count > 0)", "type": "value",
             "depends_on": {"count": lambda v: (v or 0) > 0},
             "param_info": {'dtype': float, 'unit': 'nm', 'latex_name': 's', 'default': 100.0}},
    "enabled": {"title": "Enabled", "type": "bool", "param_info": {'default': True}},
    "kind": {"title": "Kind", "type": "option", "param_info": {'options_list': ['A', 'B', 'C']}},
}
DEMO_SECTION_KEY = "demo-section"
DEMO_DEFAULT_CONFIG = {DEMO_SECTION_KEY: {"count": 2, "size": 100.0, "enabled": True, "kind": "A"}}


def _column(title, subtitle, widget):
    """One labelled column: a bold title, an italic description, then the demonstrated widget."""
    col = QWidget()
    lay = QVBoxLayout(col)
    t = QLabel(title)
    t.setStyleSheet("font-weight: bold; font-size: 13pt;")
    s = QLabel(subtitle)
    s.setWordWrap(True)
    s.setStyleSheet(f"color: gray; font-style: italic; font-size: {settings.FontSize.SMALL}pt;")
    lay.addWidget(t)
    lay.addWidget(s)
    lay.addWidget(widget)
    lay.addStretch()
    return col


def _vline():
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _build_level1():
    """Level 1: individual SimpleParameterWidgets, one per supported type, no TOML backing."""
    def demo_callback(widget, update_cache_fn, load_toml_fn, config_path):
        # the extra_button callback signature is (widget, update_cache_fn, load_toml_fn, config_path)
        print(f"[level 1] extra button clicked; current value = {widget.param_value}")

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(SimpleParameterWidget(
        title="A numeric value", type="value",
        param_info={'dtype': float, 'unit': 'nm', 'latex_name': '\\delta', 'default': 1.0}))
    lay.addWidget(SimpleParameterWidget(
        title="A boolean", type="bool", param_info={'default': True}))
    lay.addWidget(SimpleParameterWidget(
        title="An option", type="option",
        param_info={'options_list': ['first', 'second', 'third']}))
    lay.addWidget(SimpleParameterWidget(
        title="A value with an extra button", type="value",
        param_info={'dtype': int, 'unit': '', 'latex_name': 'n', 'default': 10},
        extra_button={'label': 'Estimate', 'tooltip': 'demo callback', 'callback': demo_callback}))
    return box


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    # one throwaway TOML shared by levels 2 and 3 (printed so you can open it):
    tmp_toml = Path(tempfile.mkdtemp()) / "demo_cache.toml"
    print("common/gui/base walkthrough — demo cache TOML:", tmp_toml)
    update_cache = make_update_cache(tmp_toml, DEMO_DEFAULT_CONFIG)

    # level 2: bare section (no frame) built from the demo dict
    level2 = BaseSectionWidget(
        params_ui_dict=DEMO_SECTION_UI,
        toml_section_key=DEMO_SECTION_KEY,
        update_cache_fn=update_cache,
        load_toml_fn=load_or_create_toml,
        config_path=tmp_toml,
    )
    # level 3: the same dict, wrapped by BaseSectionQGroup in a titled, bordered frame
    level3 = BaseSectionQGroup(
        parent=None,
        update_cache_fn=update_cache,
        load_toml_fn=load_or_create_toml,
        title="BaseSectionQGroup (this titled frame is what it adds)",
        params_ui_dict=DEMO_SECTION_UI,
        toml_section_key=DEMO_SECTION_KEY,
        config_path=tmp_toml,
    )

    window = QWidget()
    window.setWindowTitle("common/gui/base — architecture walkthrough (3 levels)")
    outer = QVBoxLayout(window)

    banner = QLabel(
        "common/gui/base — parameter-UI hierarchy (see __init__.py).\n"
        "Left to right, each level composes the previous one. Edit 'count' to 0 in "
        "levels 2 & 3 to see the 'size' row hide (depends_on)."
    )
    banner.setWordWrap(True)
    banner.setStyleSheet("font-size: 12pt;")
    outer.addWidget(banner)

    columns = QHBoxLayout()
    columns.addWidget(_column(
        "1. SimpleParameterWidget",
        "The atom: one parameter dict -> one line. Shown standalone (no TOML). "
        "Types: value / bool / option, plus an optional extra button.",
        _build_level1()))
    columns.addWidget(_vline())
    columns.addWidget(_column(
        "2. BaseSectionWidget",
        "Composition: a whole UI dict -> stacked atoms with separators, TOML-backed, "
        "with depends_on. Bare — NO frame, NO title.",
        level2))
    columns.addWidget(_vline())
    columns.addWidget(_column(
        "3. BaseSectionQGroup",
        "Visual grouping: the exact same section as level 2, wrapped in a titled, "
        "bordered QGroupBox. The frame is all it adds.",
        level3))
    outer.addLayout(columns)

    window.resize(1150, 380)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
