"""
Abstract base class for all iterative algorithms.

Subclasses implement run(g, H, params) -> f.
Everything else (threading, communication, interruption) is provided here.
"""

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
    pipeline through three function attributes wired by BasePipeline.run():
        _print(msg)         -> pipeline._print
        _on_finished(f)     -> pipeline._on_algo_finished
        _on_error(err)      -> pipeline._on_algo_error
    """

    # ── class attributes ──────────────────────────────────────────────────

    name = ""
    ui_params = {}
    estimator_type = ""         # "MAP" or "MMSE"
    uses_denoiser = False
    uses_regularization = False
    supported_features = set()  # e.g. {"2d"}, {"3d", "anisotropic"}

    # ── class methods ─────────────────────────────────────────────────────

    ## assembles docstrings from the full inheritance chain
    @classmethod
    def description(cls) -> str:
        parts = [f"[{cls.name}] ({cls.estimator_type} estimator)"]
        for klass in reversed(cls.__mro__):
            if klass.__doc__ and klass not in (object, ABC):
                parts.append(klass.__doc__.strip())
        parts.append(f"Uses denoiser: {cls.uses_denoiser}")
        parts.append(f"Uses regularization: {cls.uses_regularization}")
        return "\n\n".join(parts)

    ## returns ui_params filtered by the intersection of problem and algo features
    @classmethod
    def get_ui_params(cls, problem_features=None) -> dict:
        active = (problem_features & cls.supported_features
                  if problem_features else cls.supported_features)
        return {
            key: param for key, param in cls.ui_params.items()
            if param.get("requires", set()).issubset(active)
        }

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(self):
        self.active_features = set()
        self._stop_event = threading.Event()
        self._thread = None
        self._latest_f = None
        ## function attributes wired by BasePipeline.run()
        self._print = lambda msg: None
        self._on_finished = lambda f: None
        self._on_error = lambda err: None

    # ── threading ─────────────────────────────────────────────────────────

    ## stores a snapshot of f for live preview (polled by the UI timer)
    def _update_figure(self, f):
        self._latest_f = f.detach().clone()

    ## launches run() in a background thread
    def _run(self, g, H, params: Dict[str, Any]):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_wrapper,
            args=(g, H, params),
            daemon=True
        )
        self._thread.start()

    ## requests interruption and waits for the thread to join
    def stop_running(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._print("Algorithm interruption requested")
            self._thread.join(timeout=3.0)

    ## returns True if stop was requested (check this inside the run() loop)
    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    ## wraps run() to catch exceptions and call _on_finished or _on_error
    def _run_wrapper(self, g, H, params):
        try:
            result = self.run(g, H, params)
            if not self._stop_event.is_set():
                self._on_finished(result)
        except Exception as e:
            traceback.print_exc()
            self._on_error(str(e))

    ## seeds all RNGs for reproducibility
    @staticmethod
    def fix_randomness(seed=123):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── factory helpers ───────────────────────────────────────────────────
    # Auto-create diff_ops, data_fidelity, regularization, loss_computer
    # from params dict. Used by base algo classes so that concrete
    # subclasses only provide apply_forward/apply_adjoint/supported_features.

    ## warns if a 2D-only denoiser will be applied slice-by-slice on 3D data
    def _warn_if_slice_by_slice(self, denoiser_name, data):
        from common.denoisers import DENOISER_REGISTRY
        if denoiser_name == "None":
            return
        denoiser = DENOISER_REGISTRY.get(denoiser_name)
        if denoiser is None:
            return
        is_3d = data.dim() >= 3 and data.shape[0] > 1
        if is_3d and not denoiser.supports_3d:
            self._print(
                f"[WARNING] Denoiser '{denoiser_name}' does not support 3D natively. "
                f"It will be applied slice-by-slice along the Z axis."
            )

    ## creates a DifferentialOperators with delta from params
    def _create_diff_ops(self, params):
        from common.algorithms.base import DifferentialOperators
        return DifferentialOperators(delta=params.get('delta', 1.0))

    ## creates a DataFidelity instance from params
    def _create_data_fidelity(self, params):
        from common.algorithms.reusable.data_fidelities import DATA_FIDELITY_REGISTRY, GaussianFidelity
        key = params.get('data_fidelity', GaussianFidelity.display_name)
        cls = DATA_FIDELITY_REGISTRY.get(key, GaussianFidelity)
        return cls(**self._extract_init_kwargs(cls, params))

    ## creates a Regularization instance from params
    def _create_regularization(self, params):
        from common.algorithms.reusable.regularizations import REGULARIZATION_REGISTRY, NoRegularization
        if not self.uses_regularization:
            return NoRegularization()
        key = params.get('reg', NoRegularization.display_name)
        cls = REGULARIZATION_REGISTRY.get(key, NoRegularization)
        return cls(**self._extract_init_kwargs(cls, params))

    ## creates a fully configured LossComputer from params
    def _create_loss_computer(self, g, H, params):
        from common.algorithms.base.loss_computer import LossComputer
        diff_ops = self._create_diff_ops(params)
        data_fidelity = self._create_data_fidelity(params)
        regularization = self._create_regularization(params)
        lambda_reg = params.get('lambda_reg', 0.)
        return LossComputer(
            g, H, self.apply_forward, data_fidelity,
            regularization, diff_ops, lambda_reg
        )

    @staticmethod
    def _extract_init_kwargs(cls, params):
        from common.utils import extract_init_kwargs
        return extract_init_kwargs(cls, params)

    # ── abstract interface ────────────────────────────────────────────────

    def apply_forward(self, H, f):
        """Computes Hf. Must be overridden by concrete subclasses."""
        raise NotImplementedError

    def apply_adjoint(self, H, x):
        """Computes H^T x. Must be overridden by concrete subclasses."""
        raise NotImplementedError

    @abstractmethod
    def run(self, g, H, params: Dict[str, Any]):
        """The iterative algorithm itself. Returns the reconstructed f tensor."""
