"""
Launching an inverse problem — the same launcher for every one of them.

`run(problem, argv)` takes an `InverseProblem` and does the rest: it builds the control
window from the problem's own ui declaration, or runs a reconstruction headless. A problem
contributes no launcher code, which is why there is no `main.py` under `problems/` any more.

    matirf gui                     -> open_gui(problem)
    matirf cli -c cfg -o out/      -> run_headless(problem, ...)
"""

import argparse
import sys
import time
from pathlib import Path

from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QStyleFactory

import settings as settings
from gui.errors import install_error_handlers


def open_gui(problem) -> None:
    """Open this problem's control window, centred on the screen the cursor is on."""
    from gui.factory import control_window_class

    app = QApplication(sys.argv)
    install_error_handlers()                  # an error prints its traceback, never aborts
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette
    app.setPalette(palette())

    window = control_window_class(problem)()
    _centre(window)
    window.show()
    sys.exit(app.exec_())


def run_headless(problem, config_path: Path, output_path: Path) -> None:
    """
    Run one reconstruction with no interface, printing progress and saving the result.

    This is the whole of `<problem> cli`: build the pipeline, wire the callbacks to print,
    start, wait. It is also the shortest example of driving the framework from a script.
    """
    from fileio import load_or_create_toml
    from pipeline import pipeline_for

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = load_or_create_toml(config_path, problem.default_config)

    pipeline = pipeline_for(problem).create(config)
    pipeline.on_message = print
    pipeline.on_error = lambda error: print("[ERROR]", error)
    pipeline.on_finished = lambda result: (
        pipeline.save_results(str(output_path)),
        print(f"[INFO] Saved to {output_path}"),
    )

    print(f"[INFO] Starting the {problem.name} pipeline...")
    pipeline.start()
    while pipeline.is_running:
        time.sleep(0.05)
    pipeline_for(problem).remove(pipeline)


def run(problem, argv=None) -> None:
    """Parse `<problem> gui` / `<problem> cli -c ... -o ...` and dispatch."""
    parser = argparse.ArgumentParser(description=problem.description or problem.name)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("gui")
    cli = modes.add_parser("cli")
    cli.add_argument("-c", "--config", type=Path, required=True)
    cli.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.mode == "gui":
        open_gui(problem)
    else:
        run_headless(problem, args.config, args.output)


def _centre(window) -> None:
    """Centre on whichever screen the cursor is on — not always the primary one."""
    screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
    frame = window.frameGeometry()
    frame.moveCenter(screen.availableGeometry().center())
    window.move(frame.topLeft())
