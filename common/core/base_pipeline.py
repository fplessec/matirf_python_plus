"""
Base pipeline for any inverse problem (matirf, deconv, etc.).

Subclass BasePipeline and set 7 class attributes — no methods to override.
See class docstring for the list.
"""

from typing import Optional
import os
from os import makedirs
from os.path import join

from .enums import DataMode, PipelineState
from .base_result import BaseResult
from common.in_out import save_toml, save_txt, load_txt, save_json, load_json


class BasePipeline:
    """
    Fully concrete base class for inverse problem pipelines.

    Class attributes (set by subclasses):
        RESULT_CLASS, PROBLEM_FEATURES, ALGORITHM_REGISTRY,
        PIPELINE_OPERATIONS, SAVE_IMAGE, LOAD_IMAGE, IMAGE_EXTENSION
    """

    # ── class attributes ──────────────────────────────────────────────────

    RESULT_CLASS = None
    PROBLEM_FEATURES = None
    ALGORITHM_REGISTRY = None
    PIPELINE_OPERATIONS = None
    SAVE_IMAGE = None
    LOAD_IMAGE = None
    IMAGE_EXTENSION = None

    # ── registry ──────────────────────────────────────────────────────────
    # Each subclass gets its own _pipelines list, allowing multiple
    # reconstructions in parallel within a single session.

    _pipelines = []

    def __init_subclass__(cls, **kwargs):
        ## called once at class definition (import time), not at instantiation
        super().__init_subclass__(**kwargs)
        cls._pipelines = []
        for attr in ('RESULT_CLASS', 'PROBLEM_FEATURES', 'ALGORITHM_REGISTRY',
                     'PIPELINE_OPERATIONS', 'SAVE_IMAGE', 'LOAD_IMAGE', 'IMAGE_EXTENSION'):
            if getattr(cls, attr) is None:
                raise TypeError(f"{cls.__name__} must define class attribute {attr}")
        cls.SAVE_IMAGE = staticmethod(cls.SAVE_IMAGE)
        cls.LOAD_IMAGE = staticmethod(cls.LOAD_IMAGE)

    ## instantiates a new pipeline and registers it
    @classmethod
    def create(cls, config) -> "BasePipeline":
        pipeline = cls(config)
        cls._pipelines.append(pipeline)
        return pipeline

    ## unregisters a pipeline
    @classmethod
    def remove(cls, pipeline) -> None:
        if pipeline in cls._pipelines:
            cls._pipelines.remove(pipeline)

    ## stops all running pipelines and clears the registry
    @classmethod
    def stop_all(cls) -> None:
        for p in cls._pipelines[:]:
            p.stop()
        cls._pipelines.clear()

    @classmethod
    def get_all(cls) -> list:
        return cls._pipelines

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(self, config):
        self.config = config
        ## callbacks — wired by GUI (BaseDisplayWindow) or CLI (app.py)
        self.on_message = None
        self.on_finished = None
        self.on_error = None
        self.on_state_changed = None
        ## pipeline state
        self.result: Optional[BaseResult] = self.RESULT_CLASS()
        self.mode = DataMode.from_config(config['input-paths']['mode'])
        self.state = PipelineState.IDLE
        self.algorithm = None
        self._running = False

    # ── properties ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_synthetic_data(self) -> bool:
        return self.mode == DataMode.SYNTHETIC

    # ── communication ─────────────────────────────────────────────────────

    ## logs a message in result.messages and notifies the listener
    def _print(self, msg: str):
        self.result.messages += msg + "\n"
        if self.on_message:
            self.on_message(msg)

    ## updates pipeline state and notifies the listener
    def _set_state(self, new_state: PipelineState, do_print=True):
        old_state = self.state
        if new_state == old_state:
            return
        self.state = new_state
        if do_print:
            self._print(f"[state] {old_state.value} → {new_state.value}")
        if self.on_state_changed:
            self.on_state_changed(old_state, new_state)

    # ── life cycle ────────────────────────────────────────────────────────
    #
    # start()
    #   ├─ validate_config()
    #   ├─ setup()          builds g, H, instantiates algo
    #   ├─ run()            wires algo callbacks, launches thread
    #   │    ├─ _on_algo_finished()  → COMPLETED
    #   │    └─ _on_algo_error()     → FAILED
    #   └─ stop()           interrupts the algo thread

    ## validates config, loads data, launches the algo
    def start(self) -> None:
        self._print("Initializing run with the current configuration.")
        try:
            errors = self.PIPELINE_OPERATIONS.validate_config(self.config)
            if errors:
                self._print(f"⚠ Configuration incomplete ({len(errors)} issue(s)):")
                for err in errors:
                    self._print(f"  - {err}")
                if self.on_error:
                    self.on_error("Cannot start: please fix the configuration above.")
                return
            self._set_state(PipelineState.LOADING)
            self.setup()
            self._set_state(PipelineState.COMPUTING)
            self._running = True
            self.run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._running = False
            self._set_state(PipelineState.FAILED)
            if self.on_error:
                self.on_error(f"{type(e).__name__}: {e}")

    ## builds g and H via PipelineOperations, then instantiates the chosen algo
    def setup(self) -> None:
        if self.is_synthetic_data:
            g_synth, H_synth, f_true = self.PIPELINE_OPERATIONS.build_g_H_synthetic(self.config)
            self.result.g = g_synth
            self.result.H = H_synth
            self.result.f_true = f_true
        else:
            g, H = self.PIPELINE_OPERATIONS.build_g_H_real(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = None
        self._create_algorithm()

    ## instantiates the selected algorithm and sets its active features
    def _create_algorithm(self) -> None:
        algo_name = self.config['algorithm']
        algo_class = self.ALGORITHM_REGISTRY[algo_name]
        self.algorithm = algo_class()
        self.algorithm.active_features = self.PROBLEM_FEATURES & algo_class.supported_features

    def run(self) -> None:
        ## wires the algorithm's function attributes to the pipeline
        self.algorithm._print = self._print
        self.algorithm._on_finished = self._on_algo_finished
        self.algorithm._on_error = self._on_algo_error
        self.algorithm._run(self.result.g, self.result.H, self.config['algo-params'])

    ## called by the algo thread on convergence: stores f, computes metrics, notifies
    def _on_algo_finished(self, f):
        self.result.f = f
        if self.mode == DataMode.SYNTHETIC:
            self.PIPELINE_OPERATIONS.compute_synthetic_outputs(
                self.result, self.config, self.PROBLEM_FEATURES
            )
        self._running = False
        self._set_state(PipelineState.COMPLETED)
        if self.on_finished:
            self.on_finished(self.result)

    ## called by the algo thread on exception
    def _on_algo_error(self, err):
        self._running = False
        self._set_state(PipelineState.FAILED)
        if self.on_error:
            self.on_error(err)

    ## requests the algo to stop and updates state
    def stop(self) -> None:
        if self.algorithm:
            self.algorithm.stop_running()
        self._running = False
        if self.state in (PipelineState.LOADING, PipelineState.COMPUTING):
            self._set_state(PipelineState.INTERRUPTED)
            self._print("Algorithm interrupted by user.")

    # ── I/O ───────────────────────────────────────────────────────────────

    ## saves f, config, messages (and f_true + metrics in synthetic mode) to disk
    def save_results(self, save_dir):
        ext = self.IMAGE_EXTENSION
        makedirs(save_dir, exist_ok=True)
        if self.result.f is None:
            raise ValueError("Cannot save results: 'f' is None")
        self.SAVE_IMAGE(self.result.f, join(save_dir, f'f.{ext}'))
        save_toml(self.config, join(save_dir, 'config.toml'))
        save_txt(self.result.messages, join(save_dir, 'messages.txt'))
        if self.mode == DataMode.SYNTHETIC:
            if self.result.f_true is None:
                raise ValueError("Synthetic mode but 'f_true' is None")
            self.SAVE_IMAGE(self.result.f_true, join(save_dir, f'f_true.{ext}'))
            if self.result.metrics is not None:
                save_json(self.result.metrics, join(save_dir, 'metrics.json'))
        self._print(f"\nSaved reconstruction in {save_dir}")

    ## loads a previously saved reconstruction from disk
    def load_results(self, directory):
        ext = self.IMAGE_EXTENSION
        required = ["config.toml", f"f.{ext}", "messages.txt"]
        for file in required:
            if not os.path.exists(join(directory, file)):
                raise FileNotFoundError(file)
        self.result = self.RESULT_CLASS(
            f=self.LOAD_IMAGE(join(directory, f"f.{ext}")),
            messages=load_txt(join(directory, "messages.txt")),
        )
        if self.mode == DataMode.SYNTHETIC:
            self.result.f_true = self.LOAD_IMAGE(join(directory, f"f_true.{ext}"))
            json_path = join(directory, 'metrics.json')
            if os.path.exists(json_path):
                self.result.metrics = load_json(json_path)
            if self.result.f is not None and self.result.f_true is not None:
                self.PIPELINE_OPERATIONS.compute_synthetic_outputs(
                    self.result, self.config, self.PROBLEM_FEATURES
                )
        self._print(f"\nLoaded reconstruction from {directory}")
        self._set_state(PipelineState.COMPLETED, do_print=False)
