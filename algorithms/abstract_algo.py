import threading
import traceback
from typing import Dict, Any
from abc import ABC, abstractmethod
from PyQt5.QtCore import QObject, pyqtSignal


class AlgorithmSignals(QObject):
    # to handle the signals safely
    message = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class Algorithm(ABC):
    def __init__(self, measurement_params=None, oper_params=None):
        self.window = None   # <- this object is a DisplayWindow when using the GUI
        self._stop_event = threading.Event()
        self._thread = None
        self.signals = AlgorithmSignals()
        self.measurement_params = measurement_params
        self.oper_params = oper_params

    def _run(self, g, H, params: Dict[str, Any], window=None):
        # _run is a wrapper for run, making it interruptible
        self.window = window
        self._stop_event.clear()
        if self.window is not None:
            self.signals.message.connect(self.window._print)
            self.signals.finished.connect(self._on_finished)
            self.signals.error.connect(self._on_error)
        else:
            self.signals.message.connect(lambda msg: print(msg))
            self.signals.finished.connect(lambda result: print("Algorithm finished."))
            self.signals.error.connect(lambda error: print(f"Exception during algorithm's execution: {error}"))
        self._thread = threading.Thread(
            target=self._run_wrapper,
            args=(g, H, params)
        )
        self._thread.start()

    def _run_wrapper(self, g, H, params: Dict[str, Any]):
        # internal wrapper that periodically checks if an interruption has been requested or is there's an error
        try:
            result = self.run(g, H, params)
            if not self._stop_event.is_set():
                self.signals.finished.emit(result)
        except Exception as e:
            print("=== Exception Traceback ===")
            traceback.print_exc()
            print("=====================")
            self.signals.error.emit(str(e))

    def _on_finished(self, result):
        # is called when the algorithm finish successfully
        if self.window is not None:
            self.window.update_f(result)
            self.window.update_plot()

    def _on_error(self, error_msg):
        # is called if there's an exception when the algorithm is running
        self._print(f"Exception during execution: {error_msg}")
        self._print(f"-> See on terminal console the complete traceback.")

    @abstractmethod
    def run(self, g, H, params: Dict[str, Any]):
        # to be implemented for each subclass, ie main function for each algorithm
        pass

    def _print(self, string):
        # thread safe method to execute 'self.window._print()' ie printing in the DisplayWindow
        self.signals.message.emit(string)

    def stop_running(self):
        # to safely stop the algorithm
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._print("Algorithm interruption requested")
            # wait that the thread is killed (max 3 secondes)
            self._thread.join(timeout=3.0)

    def is_stop_requested(self):
        # when called in the 'run' method, checks if an interruption has been requested
        # very important method that should be call at the start of each iteration of any algorithm in the 'run' method
        return self._stop_event.is_set()