"""
Classe abstraite pour tous les algorithmes de déconvolution 2D.

Différences avec algorithms/abstract_algo.py de matirf :
    - pas de measurement_params / oper_params dans __init__ : en deconv 2D
      l'algorithme n'a besoin que de g, H et de ses propres hyperparamètres
      (passés via params dans _run). Toute l'info "physique" est déjà encapsulée
      dans la PSF H.
"""

import random
import threading
import traceback
from typing import Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import torch


class Algorithm(ABC):
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

        # callbacks python (pas de Qt à ce niveau)
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
        """Lance l'algorithme dans un thread d'arrière-plan."""
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
