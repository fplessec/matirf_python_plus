"""
Shared application launcher for any inverse problem.

Provides:
    run_app(control_window_class, pipeline_class, load_config_fn, name)
        Parses CLI arguments and dispatches to GUI or CLI mode.
"""

import sys
import argparse
import time
from pathlib import Path

from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QStyleFactory

import common.settings as settings


def _center_window(window):
    cursor_pos = QCursor.pos()
    screen = QApplication.screenAt(cursor_pos)
    if screen is None:
        screen = QApplication.primaryScreen()
    geo = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(geo.center())
    window.move(frame.topLeft())


def _open_gui(control_window_class):
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette
    app.setPalette(palette())
    window = control_window_class()
    _center_window(window)
    window.show()
    sys.exit(app.exec_())


def _run_cli(config_path, output_path, pipeline_class, load_config_fn):
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_config_fn(config_path)
    print("[INFO] Starting pipeline...")

    def on_message(msg):
        print(msg)

    def on_finished(result):
        print("[INFO] Reconstruction finished")
        pipeline.save_results(str(output_path))
        print(f"[INFO] Saved to {output_path}")

    def on_error(err):
        print("[ERROR]", err)

    pipeline = pipeline_class.create(config)
    pipeline.on_message = on_message
    pipeline.on_finished = on_finished
    pipeline.on_error = on_error
    pipeline.start()
    while pipeline.is_running:
        time.sleep(0.05)
    pipeline_class.remove(pipeline)


def run_app(control_window_class, pipeline_class, load_config_fn, name=""):
    """
    Shared entry point for any inverse problem.

    Parameters
    ----------
    control_window_class : type
        The QMainWindow subclass for this problem's GUI.
    pipeline_class : type
        The BasePipeline subclass for this problem.
    load_config_fn : callable
        Function to load a TOML config file (path -> dict).
    name : str
        Description for argparse help text.
    """
    parser = argparse.ArgumentParser(description=name)
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("gui")

    cli = sub.add_parser("cli")
    cli.add_argument("-c", "--config", type=Path, required=True)
    cli.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "gui":
        _open_gui(control_window_class)
    else:
        _run_cli(args.config, args.output, pipeline_class, load_config_fn)
