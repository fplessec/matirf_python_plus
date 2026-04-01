# import sys
# import argparse
# from pathlib import Path
#
# from PyQt5.QtGui import QCursor
# from PyQt5.QtWidgets import QStyleFactory, QApplication
#
# from gui import ControlWindow
# import settings
# from in_out import load_or_create_toml
#
#
# def center_window_on_cursor_screen(window):
#     """
#     Function that allows the window to be opened in the same screen where the user's mouse is (useful if the user
#     has more than 1 screen, anyway... small detail but convenient).
#     """
#     cursor_pos = QCursor.pos()
#     screen = QApplication.screenAt(cursor_pos)
#     if screen is None:
#         screen = QApplication.primaryScreen()
#     screen_geometry = screen.availableGeometry()
#     window_geometry = window.frameGeometry()
#     window_geometry.moveCenter(screen_geometry.center())
#     window.move(window_geometry.topLeft())
#
# # GUI MODE
# def open_gui():
#     app = QApplication(sys.argv)
#
#     app_style = QStyleFactory.create(settings.app_style)
#     app.setStyle(app_style)
#     app_palette = settings.dark_palette if settings.dark_style else settings.light_palette
#     app.setPalette(app_palette())
#
#     CW = ControlWindow()
#     center_window_on_cursor_screen(CW)
#     CW.show()
#
#     sys.exit(app.exec_())
#
# #
# # def main_cli():
# #     config = load_config_from_args()
# #
# #     from core.pipeline_manager import PipelineManager
# #
# #     def print_msg(msg):
# #         print(msg)
# #
# #     def finished(result):
# #         print("Reconstruction terminée")
# #
# #     pipeline = PipelineManager.create(
# #         config,
# #         callbacks={
# #             "message": print_msg,
# #             "finished": finished,
# #             "error": lambda e: print("Error:", e)
# #         }
# #     )
# #
# #     pipeline.start()
#
#
# # TERMINAL MODE
# def run_terminal(config_path: Path, output_path: Path):
#     """
#     Run the application in terminal mode.
#
#     Parameters
#     ----------
#     config_path : Path
#         Path to the TOML configuration file.
#     output_path : Path
#         Path to the output directory (created at the end).
#     """
#
#     # ---- Validation
#     if not config_path.exists():
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#
#     # ---- Load config (example)
#     print(f"[INFO] Loading config from: {config_path}")
#
#     config = load_or_create_toml(config_path)
#     print(config)
#
#     # ---- Main processing (placeholder)
#     print("[INFO] Running processing...")
#     # Exemple:
#     # result = process(config)
#
#
#
#
#     # ---- Save results
#     print(f"[INFO] Creating output directory: {output_path}")
#     output_path.mkdir(parents=True, exist_ok=True)
#
#     # TODO: sauvegarder tes résultats ici
#     # save_tif(output_path / "result.tif", result)
#
#     print("[INFO] Done.")
#
#
# def main():
#
#     parser = argparse.ArgumentParser(description="Implémentation seulement avec un GUI pour l'instant")
#     subparsers = parser.add_subparsers(dest="mode", required=True)
#
#     # ---- GUI subcommand
#     subparsers.add_parser("gui", help="Launch GUI")
#
#     # ---- TERMINAL subcommand
#     terminal_parser = subparsers.add_parser("terminal", help="Run in terminal")
#     terminal_parser.add_argument(
#         "-c", "--config",
#         dest="config_path",
#         type=Path,
#         required=True,
#         help="Path to config.toml file",
#     )
#     terminal_parser.add_argument(
#         "-o", "--output",
#         dest="output_path",
#         type=Path,
#         required=True,
#         help="Path to output directory",
#     )
#
#
#     args = parser.parse_args()
#     if args.mode == 'gui':
#         open_gui()
#     elif args.mode == 'terminal':
#         run_terminal(
#             config_path=args.config_path,
#             output_path=args.output_path,
#         )
#
#
# if __name__ == '__main__':
#     main()




# main.py



import sys
import argparse
from pathlib import Path
import time
T0 = time.time()

from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QStyleFactory

import settings
from gui.control_window import ControlWindow
from in_out import load_or_create_toml
from core import PipelineManager



# =========================
# GUI
# =========================
def center_window(window):
    cursor_pos = QCursor.pos()
    screen = QApplication.screenAt(cursor_pos)
    if screen is None:
        screen = QApplication.primaryScreen()

    geo = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(geo.center())
    window.move(frame.topLeft())


def open_gui():

    app = QApplication(sys.argv)

    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette
    app.setPalette(palette())

    window = ControlWindow()
    center_window(window)
    window.show()

    sys.exit(app.exec_())


# =========================
# CLI
# =========================
def run_client(config_path: Path, output_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_or_create_toml(config_path)

    print("[INFO] Starting pipeline...")

    # =========================
    # CALLBACKS CLI
    # =========================
    def on_message(msg):
        print(msg)

    def on_finished(result):
        print("[INFO] Reconstruction finished")

        pipeline.save_results(str(output_path))

        print(f"[INFO] Saved to {output_path}")

    def on_error(err):
        print("[ERROR]", err)

    # =========================
    # PIPELINE
    # =========================
    pipeline = PipelineManager.create(
        config,
        callbacks={
            "message": on_message,
            "finished": on_finished,
            "error": on_error,
        }
    )
    pipeline.start()
    # attendre la fin
    while pipeline._running:
        time.sleep(0.05)
    PipelineManager.remove(pipeline)


# =========================
# ENTRY POINT
# =========================
def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("gui")

    cli = sub.add_parser("cli")
    cli.add_argument("-c", "--config", type=Path, required=True)
    cli.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "gui":
        open_gui()
    else:
        run_client(args.config, args.output)


if __name__ == "__main__":
    main()