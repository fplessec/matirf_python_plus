import sys
import argparse
import time
from pathlib import Path

## adds the project root to sys.path so absolute imports (deconv.*, base.*, etc.) work:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QStyleFactory

import settings
from deconv.gui import DeconvControlWindow
from deconv.in_out import load_or_create_toml
from deconv.core import DeconvPipelineManager


## centers the window on the screen where the cursor is:
def center_window(window):
    cursor_pos = QCursor.pos()
    screen = QApplication.screenAt(cursor_pos)
    if screen is None:
        screen = QApplication.primaryScreen()
    geo = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(geo.center())
    window.move(frame.topLeft())


## launches the deconv GUI:
def open_gui():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette
    app.setPalette(palette())
    window = DeconvControlWindow()
    center_window(window)
    window.show()
    sys.exit(app.exec_())


## runs the deconv pipeline from a TOML config and saves results to output_path:
def run_cli(config_path: Path, output_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_or_create_toml(config_path)
    print("[INFO] Starting deconv pipeline...")

    def on_message(msg):
        print(msg)

    def on_finished(result):
        print("[INFO] Deconvolution finished")
        pipeline.save_results(str(output_path))
        print(f"[INFO] Saved to {output_path}")

    def on_error(err):
        print("[ERROR]", err)

    pipeline = DeconvPipelineManager.create(
        config,
        callbacks={
            "message": on_message,
            "finished": on_finished,
            "error": on_error,
        }
    )
    pipeline.start()

    while pipeline._running:
        time.sleep(0.05)
    DeconvPipelineManager.remove(pipeline)


## parses CLI args and dispatches to gui or cli mode:
def main():
    parser = argparse.ArgumentParser(description="2D Deconvolution")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("gui")

    cli = sub.add_parser("cli")
    cli.add_argument("-c", "--config", type=Path, required=True)
    cli.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "gui":
        open_gui()
    else:
        run_cli(args.config, args.output)


if __name__ == "__main__":
    main()
