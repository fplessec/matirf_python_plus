"""
What happens when something goes wrong while the interface is running.

Without this module, an exception raised in a click, a value change or a timer — any Python
function Qt calls — makes PyQt5 abort the whole process (PyQt >= 5.5 calls qFatal when the
default sys.excepthook is in place): the window disappears, macOS reports "Python quit
unexpectedly", and the traceback is lost.

`install_error_handlers()` changes that, once, for every window of the application:

    sys.excepthook         an exception in the GUI thread prints its full traceback to the
                           terminal and, optionally, opens a short dialog; the interface
                           KEEPS RUNNING (possibly in an inconsistent state — hence the dialog)
    threading.excepthook   the same for background threads (e.g. the solver's), minus the
                           dialog, which only the GUI thread may open
    faulthandler           a real low-level crash (a segfault or abort inside Qt or torch)
                           cannot be caught; Python at least prints where it happened

Every graphical entry point calls it right after creating the QApplication: `matirf gui`
(gui/app.py), `matirf synth` (problems/matirf/synthetic/gui.py), `settings gui`
(settings/gui.py). The GUI smoke test uses the same function to count such errors as
failures instead of dying silently.
"""

import faulthandler
import sys
import threading
import traceback

_installed = False
_dialog_open = False


def install_error_handlers(show_dialog: bool = True, on_error=None) -> None:
    """
    Route uncaught exceptions to the terminal (and a dialog) instead of aborting.

    >> show_dialog : bool      open a small dialog for errors in the GUI thread
    >> on_error    : callable  optional hook, called with (exc_type, exc_value, exc_tb)
                               for every uncaught error — how the smoke test counts them
    """
    global _installed
    faulthandler.enable()

    def gui_thread_hook(exc_type, exc_value, exc_tb):
        _report(exc_type, exc_value, exc_tb, where="the interface")
        if on_error is not None:
            on_error(exc_type, exc_value, exc_tb)
        if show_dialog:
            _show_dialog(exc_type, exc_value)

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "a background thread"
        _report(args.exc_type, args.exc_value, args.exc_traceback, where=f"thread {name!r}")
        if on_error is not None:
            on_error(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = gui_thread_hook
    threading.excepthook = thread_hook
    _installed = True


def _report(exc_type, exc_value, exc_tb, where: str) -> None:
    print(f"\n[ERROR] Uncaught exception in {where} — the application keeps running:",
          file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()


def _show_dialog(exc_type, exc_value) -> None:
    """A short, non-recursive notice: one dialog at a time, never an error from the dialog."""
    global _dialog_open
    if _dialog_open:
        return
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is None:
            return
        _dialog_open = True
        QMessageBox.critical(
            None, "An error occurred",
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"The full traceback is printed in the terminal.\n"
            f"The application is still running, but what you were doing may not have "
            f"completed: check the current settings before continuing.")
    except Exception:
        pass
    finally:
        _dialog_open = False
