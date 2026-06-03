import random
import threading
import traceback
from typing import Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import torch


class Algorithm(ABC):
    """
    Abstract base class for all iterative algorithms (matirf and deconv).

    The algorithm runs in a background thread and communicates with the
    pipeline through three callbacks: "message", "finished", "error".

    Provides (already implemented):
        _run()              : launches run() in a background thread
        stop_running()      : requests interruption and waits for the thread to join
        is_stop_requested() : returns True if stop was requested (check inside run() loop)
        _print()            : shortcut to emit a "message" callback
        _emit()             : calls a callback by name from self.callbacks
        fixe_randomness()   : seeds all RNGs for reproducibility

    Must be implemented by each subclass:
        run(g, H, params)   : the iterative algorithm itself, returns the reconstructed f
    """
    def __init__(self):
        ## threading event used to request interruption from outside the thread:
        self._stop_event = threading.Event()
        ## reference to the background thread running the algorithm:
        self._thread = None
        ## a dictionary for the three algorithm callbacks, wired by the pipeline in run():
        self.callbacks = {}
        # { "message": ..., "finished": ..., "error": ... }
          # - "message" is called to report progress during iterations
          # - "finished" is called when run() returns successfully
          # - "error" is called when run() raises an exception

    ## _emit calls the desired algorithm callback from the dictionary self.callbacks:
    def _emit(self, name, *args):
        if name in self.callbacks:
            self.callbacks[name](*args)

    ## shortcut to emit the "message" callback:
    def _print(self, msg: str):
        self._emit("message", msg)

    ## launches run() in a background thread:
    def _run(self, g, H, params: Dict[str, Any]):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_wrapper,
            args=(g, H, params),
            daemon=True
        )
        self._thread.start()

    ## requests interruption and waits for the thread to join:
    def stop_running(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._emit("message", "Algorithm interruption requested")
            self._thread.join(timeout=3.0)

    ## returns True if stop was requested (check this inside the run() loop):
    def is_stop_requested(self):
        return self._stop_event.is_set()

    ## wraps run() to catch exceptions and emit "finished" or "error":
    def _run_wrapper(self, g, H, params):
        try:
            result = self.run(g, H, params)
            if not self._stop_event.is_set():
                self._emit("finished", result)
        except Exception as e:
            traceback.print_exc()
            self._emit("error", str(e))

    ## seeds all RNGs for reproducibility:
    @staticmethod
    def fixe_randomness(seed=123):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @abstractmethod
    def run(self, g, H, params: Dict[str, Any]):
        """The iterative algorithm itself. Returns the reconstructed f tensor."""
