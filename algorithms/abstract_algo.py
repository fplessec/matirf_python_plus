import random
import threading
import traceback
from typing import Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import torch


class Algorithm(ABC):
    def __init__(self, measurement_params, oper_params):
        self.measurement_params = measurement_params
        self.oper_params = oper_params

        self._stop_event = threading.Event()
        self._thread = None

        # callbacks python (no Qt)
        self.callbacks = {}

    # =========================
    # CALLBACK SYSTEM
    # =========================
    def _emit(self, name, *args):
        if name in self.callbacks:
            self.callbacks[name](*args)

    def _print(self, msg: str):
        self._emit("message", msg)

    # =========================
    # PUBLIC API
    # =========================
    def _run(self, g, H, params: Dict[str, Any]):
        """Run algorithm in background thread."""
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run_wrapper,
            args=(g, H, params),
            daemon=True
        )
        self._thread.start()

    def stop_running(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._emit("message", "Algorithm interruption requested")
            self._thread.join(timeout=3.0)

    def is_stop_requested(self):
        return self._stop_event.is_set()

    # =========================
    # INTERNAL
    # =========================
    def _run_wrapper(self, g, H, params):
        try:
            result = self.run(g, H, params)

            if not self._stop_event.is_set():
                self._emit("finished", result)

        except Exception as e:
            traceback.print_exc()
            self._emit("error", str(e))

    # =========================
    # UTILS
    # =========================
    @staticmethod
    def fixe_randomness(seed=123):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =========================
    # ABSTRACT
    # =========================
    @abstractmethod
    def run(self, g, H, params: Dict[str, Any]):
        pass