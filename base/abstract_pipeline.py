"""
Abstract pipeline for any inverse problem (matirf, deconv, etc.).

Contains all common logic:
    - state machine (IDLE → LOADING → COMPUTING → COMPLETED / FAILED / INTERRUPTED)
    - callback system (message, finished, error, state_changed)
    - start → validate → setup → run workflow
    - branching of algorithm callbacks
    - clean stop / interruption

The concrete subclasses implement only:
    - setup()              : load data, construct the operator, instantiate the algorithm
    - validate_config()    : verify that the configuration is complete
    - _on_algo_finished()  : post-processing after convergence (metrics, etc.)
    - save_results()       : save to disk
    - load_results()       : load from disk
"""


from abc import ABC, abstractmethod
from typing import Optional

from .enums import DataMode, PipelineState
from .base_result import BaseResult


class AbstractPipeline(ABC):
    """
    Abstract base class for inverse problem pipelines.

    Provides (already implemented):
        start()             : full workflow: validate → setup → run
        run()               : wires algorithm callbacks and launches the algorithm thread
        stop()              : requests algorithm interruption and updates state
        _set_state()        : transitions the pipeline state and notifies listeners
        _emit()             : calls a pipeline callback by name from self.callbacks
        _on_algo_message()  : relays algorithm messages to result.messages and pipeline "message" callback
        _on_algo_error()    : sets state to FAILED and emits pipeline "error"

    Must be implemented by each subclass:
        _create_result()    : returns an empty Result instance (ReconstructionResult, DeconvResult, ...)
        validate_config()   : returns a list of config error strings (empty = valid)
        setup()             : loads data, builds the operator H, instantiates the algorithm
        _on_algo_finished() : stores f, computes metrics/diff, emits pipeline "finished"
        save_results()      : persists the result to disk
        load_results()      : restores a result from disk
    """
    def __init__(self, config, callbacks=None):
        self.config = config
        ## a dictionary for the three pipeline callbacks::
        self.callbacks = callbacks or {}
        # { "message": ..., "finished": ..., "error": ...}
          # - "message" is called to track real-time progress
          # - "finished" is called when the pipeline is COMPLETED
          # - "error" is called when the pipeline is FAILED
        ## an attribute that stores all the variables used and produced by the pipeline:
        self.result: Optional[BaseResult] = self._create_result()
        ## the mode of the reconstruction (synthetic, real):
        self.mode = DataMode.from_config(config['input-paths']['mode'])
        ## the state of the pipeline:
        self.state = PipelineState.IDLE
        self.algorithm = None  # the selected algo from self.config
        self._running = False  # True while the algo thread is running


    @abstractmethod
    def _create_result(self) -> BaseResult:
        """Returns an empty instance of the concrete Result type (ReconstructionResult, DeconvResult, ...)."""

    ## state management:
    def _set_state(self, new_state: PipelineState):
        old_state = self.state
        if new_state == old_state:
            return
        self.state = new_state
        self._emit("state_changed", old_state, new_state)

    ## _emit calls the desired pipeline callback from the dictionary self.callbacks:
    def _emit(self, name, *args):
        if name in self.callbacks:
            self.callbacks[name](*args)

    ## which mode is the reconstruction:
    def is_synthetic_data(self):
        return self.mode == DataMode.SYNTHETIC

    ## to start the reconstruction pipeline:
    def start(self):
        # start()
        #   ├─ validate_config()     <- is the reconstruction config valid, if not: _set_state(FAILED)
        #   ├─ _set_state(LOADING)
        #   ├─ setup()               <- loads the data, computes the operator, instantiates the algo object
        #   ├─ _set_state(COMPUTING)
        #   └─ run()                 <- connects the algo callbacks to the algo object and launches the algo thread
        #        ├─ _on_algo_message()
        #        ├─ _on_algo_finished()   -> _set_state(COMPLETED)
        #        └─ _on_algo_error()      -> _set_state(FAILED)
        self._emit("message", "Initializing run with the current configuration.")
        try:
            errors = self.validate_config()
            if errors:
                self._emit("message", f"⚠ Configuration incomplete ({len(errors)} issue(s)):")
                for err in errors:
                    self._emit("message", f"  - {err}")
                self._emit("error", "Cannot start: please fix the configuration above.")
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
            self._emit("error", f"{type(e).__name__}: {e}")

    @abstractmethod
    def validate_config(self) -> list[str]:
        """Checks if reconstruction config is valid (complete and compliant)."""

    @abstractmethod
    def setup(self):
        """Loads the data, computes the operator, instantiates the algo."""

    def run(self):
        """Connects the algo callbacks to the algo object and launches the algo thread."""
        self.algorithm.callbacks = {
            "message": self._on_algo_message,
            "finished": self._on_algo_finished,
            "error": self._on_algo_error,
        }
        self.algorithm._run(self.result.g, self.result.H, self.config['algo-params'])

    ## algo callback to track real-time progress:
    def _on_algo_message(self, msg):
        self.result.messages += msg + "\n"
        self._emit("message", msg)

    @abstractmethod
    def _on_algo_finished(self, f):
        """Post-processing after convergence: store f, compute metrics, emit 'finished'."""

    def _on_algo_error(self, err):
        self._running = False
        self._set_state(PipelineState.FAILED)
        self._emit("error", err)

    @abstractmethod
    def save_results(self, save_dir):
        pass

    @abstractmethod
    def load_results(self, directory):
        pass

    ## to stop the pipeline:
    def stop(self):
        if self.algorithm:
            self.algorithm.stop_running()
        self._running = False
        if self.state in (PipelineState.LOADING, PipelineState.COMPUTING):
            self._set_state(PipelineState.INTERRUPTED)
            self._emit("message", "Algorithm interrupted by user.")



class BasePipelineManager:
    """
    Generic registry of running pipelines.

    Each subclass defines _pipeline_class to specify which type
    of pipeline to instantiate, and its own list of _pipelines.
    """
    _pipeline_class = None
    _pipelines = []

    @classmethod
    def create(cls, config, callbacks=None):
        pipeline = cls._pipeline_class(config, callbacks)
        cls._pipelines.append(pipeline)
        return pipeline

    @classmethod
    def remove(cls, pipeline):
        if pipeline in cls._pipelines:
            cls._pipelines.remove(pipeline)

    @classmethod
    def stop_all(cls):
        for p in cls._pipelines[:]:
            p.stop()
        cls._pipelines.clear()

    @classmethod
    def get_all(cls):
        return cls._pipelines
