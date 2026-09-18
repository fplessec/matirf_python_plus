"""
Pipeline — what actually runs a reconstruction, from a config to a saved result.

It is the COMPOSITION layer: the only module that knows about problems and solvers at the
same time. That is why it sits here at the root rather than inside `core/`. `core` declares
the contracts and deliberately imports neither side; putting the orchestration there would
have made `core` depend on `solvers`, inverting the dependency the whole design rests on.

    problems/  ──►  core  ◄──  solvers/
        │                          │
        └────────►  pipeline  ◄────┘
                        │
                       gui

One run, in order:

    validate    the problem lists every unusable setting, all at once
    prepare     the problem loads the data and builds its operator
    assemble    the objective is built from the config: noise model, prior, weight
    solve       the chosen solver minimizes it, on a background thread
    evaluate    in synthetic mode, compare against the known truth
    save        f, the config, the messages, and the metrics

Everything problem-specific happens inside `problem`; everything algorithm-specific inside
`solver`. This file contains neither, which is why it is the same file for every inverse
problem — and why adding one needs no pipeline code.
"""

import os
import traceback
from os.path import join
from typing import Optional

import torch

from core import DataMode, Objective
from core.result import Result
from core.enums import PipelineState
from fileio import save_toml, save_txt, load_txt, save_json, load_json
from solvers import SOLVERS
from solvers.differential_operators import DifferentialOperators
from solvers.fidelities import DATA_FIDELITY_REGISTRY, GaussianFidelity
from solvers.regularizers import REGULARIZATION_REGISTRY, NoRegularization


# ── assembling the objective from a parameter dict ────────────────────────────

def build_objective(prepared, params: dict, uses_regularization: bool = True) -> Objective:
    """
    Turn the user's parameter dict into the Objective to minimize.

    This is the step v1 asked each algorithm to perform for itself. Doing it once, here,
    is what lets a solver receive a finished objective and stay problem-agnostic.

    `uses_regularization` comes from the solver: a solver whose data step is a hardcoded
    quadratic (ADMM, PnP) is given no prior, because attaching one would silently suggest
    an influence it does not have.
    """
    fidelity_key = params.get("data_fidelity", GaussianFidelity.display_name)
    fidelity = DATA_FIDELITY_REGISTRY.get(fidelity_key, GaussianFidelity)()

    regularization, lambda_reg = None, 0.0
    if uses_regularization:
        reg_key = params.get("reg", NoRegularization.display_name)
        reg_class = REGULARIZATION_REGISTRY.get(reg_key, NoRegularization)
        regularization = reg_class(**_accepted_kwargs(reg_class, params))
        lambda_reg = float(params.get("lambda_reg", 0.0) or 0.0)

    ## delta defaults to 1 (isotropic); an anisotropic problem supplies the real ratio, and
    ## the operator can estimate it when the user never set it by hand:
    delta = params.get("delta")
    if delta in (None, "None", "null"):
        estimate = getattr(prepared.operator, "estimate_anisotropy_ratio", None)
        delta = estimate() if estimate is not None else 1.0
    diff_ops = DifferentialOperators(delta=float(delta))

    return Objective(prepared.operator, prepared.g, fidelity, regularization,
                     lambda_reg=lambda_reg, diff_ops=diff_ops)


def _accepted_kwargs(cls, params: dict) -> dict:
    """The subset of `params` this class's constructor actually declares."""
    from core.utils import extract_init_kwargs
    return extract_init_kwargs(cls, params)


# ── the pipeline ──────────────────────────────────────────────────────────────

class Pipeline:
    """
    One reconstruction run: a problem, a config, and the solver they select.

    ----------
    > Listening to a run :
    ----------

    Four callbacks, assigned by whoever is driving — the display window in the GUI, or a
    print in a script. They are plain attributes, not a registry, so a caller wires only
    what it cares about:

        on_message(text)                 a line of progress
        on_finished(result)              the run succeeded; `result` is complete
        on_error(text)                   the run failed or the config was rejected
        on_state_changed(old, new)       a PipelineState transition

    ----------
    > Example :
    ----------

        pipeline = Pipeline.create(MATIRF, config)
        pipeline.on_message = print
        pipeline.start()
        while pipeline.is_running:
            time.sleep(0.05)
        pipeline.save_results("results/run-01")
    """

    ## every live pipeline, so several reconstructions can run at once and be stopped together
    _running_pipelines = []

    # ── registry ─────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, problem, config: dict) -> "Pipeline":
        pipeline = cls(problem, config)
        cls._running_pipelines.append(pipeline)
        return pipeline

    @classmethod
    def remove(cls, pipeline) -> None:
        if pipeline in cls._running_pipelines:
            cls._running_pipelines.remove(pipeline)

    @classmethod
    def stop_all(cls) -> None:
        for pipeline in list(cls._running_pipelines):
            pipeline.stop()
        cls._running_pipelines.clear()

    @classmethod
    def get_all(cls) -> list:
        return list(cls._running_pipelines)

    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(self, problem, config: dict):
        self.problem = problem
        self.config = config
        self.result = Result()
        self.prepared = None
        self.solver = None
        self.state = PipelineState.IDLE
        self._running = False

        self.on_message = None
        self.on_finished = None
        self.on_error = None
        self.on_state_changed = None

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def mode(self) -> DataMode:
        return DataMode.from_config(self.config)

    @property
    def is_synthetic_data(self) -> bool:
        return self.mode is DataMode.SYNTHETIC

    @property
    def latest_preview(self) -> Optional[torch.Tensor]:
        """
        The solver's most recent iterate, for the live preview, or None.

        Polled on a timer by the display window rather than pushed: emitting a signal per
        iteration would flood the Qt event loop and freeze the interface.
        """
        return self.solver.latest if self.solver is not None else None

    def _print(self, message: str) -> None:
        """Log into the result, then notify. The result therefore carries the whole log."""
        self.result.messages += message + "\n"
        if self.on_message:
            self.on_message(message)

    def _set_state(self, new_state: PipelineState, announce: bool = True) -> None:
        if new_state == self.state:
            return
        old_state, self.state = self.state, new_state
        if announce:
            self._print(f"[state] {old_state.value} -> {new_state.value}")
        if self.on_state_changed:
            self.on_state_changed(old_state, new_state)

    # ── running ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Validate, load, assemble and launch. Returns as soon as the solver thread starts.

        Validation errors are reported as a list rather than raised one at a time, so the
        user fixes everything in one pass instead of rediscovering the next missing field
        after each attempt.
        """
        try:
            errors = self._collect_errors()
            if errors:
                self._print(f"Configuration incomplete ({len(errors)} issue(s)):")
                for error in errors:
                    self._print(f"  - {error}")
                if self.on_error:
                    self.on_error("Cannot start: please fix the configuration above.")
                return

            self._set_state(PipelineState.LOADING)
            self._print(f"Loading {self.problem.name} data ({self.mode.value}).")
            self.prepared = self.problem.prepare(self.config)
            self.result.g = self.prepared.g
            self.result.f_true = self.prepared.f_true

            solver_class = SOLVERS[self.config["algorithm"]]
            self.solver = solver_class()
            params = self.config.get("algo-params", {})
            objective = build_objective(self.prepared, params,
                                        solver_class.uses_regularization)
            self._print(f"Solver: {solver_class.name} ({solver_class.estimator_type}).")

            self.solver.on_message = self._print
            self.solver.on_finished = self._on_solved
            self.solver.on_error = self._on_failed

            self._set_state(PipelineState.COMPUTING)
            self._running = True
            self.solver.run_in_background(objective, params)

        except Exception as error:
            traceback.print_exc()
            self._running = False
            self._set_state(PipelineState.FAILED)
            if self.on_error:
                self.on_error(f"{type(error).__name__}: {error}")

    def _collect_errors(self) -> list:
        """The problem's own validation, plus the one thing it cannot know about: the solver."""
        errors = list(self.problem.validate(self.config))
        name = self.config.get("algorithm", "None")
        if name == "None":
            errors.append("Algorithm: not selected")
        elif name not in SOLVERS:
            errors.append(f"Algorithm: unknown solver {name!r} "
                          f"(available: {', '.join(SOLVERS)})")
        return errors

    def _on_solved(self, f: torch.Tensor) -> None:
        self.result.f = f
        if self.prepared.has_truth:
            self._evaluate()
        self._set_state(PipelineState.COMPLETED)
        if self.on_finished:
            self.on_finished(self.result)
        ## LAST, and deliberately so. A headless caller waits with `while pipeline.is_running`,
        ## so this flag is its exit signal: flipping it before on_finished lets the script
        ## return — and the process exit — while the callback is still saving on this daemon
        ## thread, truncating the output. Setting it here means "everything is really done".
        self._running = False

    def _on_failed(self, error: str) -> None:
        self._set_state(PipelineState.FAILED)
        if self.on_error:
            self.on_error(error)
        self._running = False       # last, for the same reason as in _on_solved

    def stop(self) -> None:
        """Ask the solver to stop at the end of its current iteration."""
        if self.solver is not None:
            self.solver.stop()
        self._running = False
        if self.state in (PipelineState.LOADING, PipelineState.COMPUTING):
            self._set_state(PipelineState.INTERRUPTED)
            self._print("Interrupted by user.")

    # ── evaluating against a known truth ─────────────────────────────────────

    def _evaluate(self) -> None:
        """
        Compare the reconstruction to the truth — meaningful only in synthetic mode.

        The optimal scale is fitted before differencing: a scale-ambiguous problem leaves f
        determined only up to a positive constant, so the raw difference would mostly show
        that constant rather than the reconstruction error.
        """
        from core.metrics import compute_all_metrics, optimal_scale

        f, f_true = self.result.f, self.result.f_true
        features = {str(feature) for feature in self.problem.features}
        extra = {}
        estimate = getattr(self.prepared.operator, "estimate_anisotropy_ratio", None)
        if estimate is not None:
            extra["delta"] = estimate()

        self.result.metrics = compute_all_metrics(f, f_true, features=features, **extra)
        alpha = optimal_scale(f, f_true)
        self.result.alpha = float(alpha)
        self.result.diff = f_true - alpha * f

    # ── persistence ──────────────────────────────────────────────────────────

    def save_results(self, directory: str) -> None:
        """
        Write everything needed to understand and reproduce this run.

            f.<ext>        the reconstruction
            config.toml    the exact parameters used
            messages.txt   the full log
            f_true.<ext>   the truth        (synthetic mode)
            metrics.json   the quality metrics (synthetic mode)
        """
        if self.result.f is None:
            raise ValueError("Nothing to save: the run produced no reconstruction.")
        os.makedirs(directory, exist_ok=True)
        extension = self.problem.image_extension

        self.problem.save_image(self.result.f, join(directory, f"f.{extension}"))
        save_toml(self.config, join(directory, "config.toml"))
        save_txt(self.result.messages, join(directory, "messages.txt"))
        if self.result.has_synthetic_truth():
            self.problem.save_image(self.result.f_true, join(directory, f"f_true.{extension}"))
            if self.result.metrics is not None:
                save_json(self.result.metrics, join(directory, "metrics.json"))
        self._print(f"Saved to {directory}")

    def load_results(self, directory: str) -> None:
        """Read a saved run back, so a finished reconstruction can be reopened and compared."""
        extension = self.problem.image_extension
        for required in ("config.toml", f"f.{extension}", "messages.txt"):
            if not os.path.exists(join(directory, required)):
                raise FileNotFoundError(join(directory, required))

        self.result = Result(
            f=self.problem.load_image(join(directory, f"f.{extension}")),
            messages=load_txt(join(directory, "messages.txt")),
        )
        truth_path = join(directory, f"f_true.{extension}")
        if os.path.exists(truth_path):
            self.result.f_true = self.problem.load_image(truth_path)
        metrics_path = join(directory, "metrics.json")
        if os.path.exists(metrics_path):
            self.result.metrics = load_json(metrics_path)
        self._print(f"Loaded from {directory}")
        self._set_state(PipelineState.COMPLETED, announce=False)

    def __repr__(self) -> str:
        return (f"<Pipeline {self.problem.name!r} {self.mode.value} "
                f"state={self.state.value}>")


## One bound class per problem, reused on every call — see `pipeline_for`.
_BOUND_PIPELINES = {}


def pipeline_for(problem) -> type:
    """
    A Pipeline subclass bound to one problem, so `create(config)` needs no problem argument.

    The GUI holds a `pipeline_class` and calls `create(config)`, `stop_all()` and
    `remove(p)` on it without knowing which problem it serves — that indirection is what
    lets one control window serve every problem. Each subclass gets its OWN registry of
    running pipelines, so closing the MA-TIRF window does not stop a deconvolution run.

    The result is MEMOIZED per problem, and that matters: the control window and the
    display window both ask for their problem's pipeline class. Returning a fresh class to
    each would give them separate registries, so a window would try to remove a pipeline
    from a registry it was never added to — the run would linger forever and never be
    stopped. One problem, one bound class.

        MatirfPipeline = pipeline_for(MATIRF)
        MatirfPipeline.create(config).start()
    """
    cached = _BOUND_PIPELINES.get(problem.name)
    if cached is not None:
        return cached
    name = f"{problem.name.capitalize()}Pipeline"
    namespace = {
        "PROBLEM": problem,
        "_running_pipelines": [],
        "__doc__": f"Pipeline bound to the {problem.name!r} inverse problem.",
    }

    def create(cls, config):
        pipeline = cls(cls.PROBLEM, config)
        cls._running_pipelines.append(pipeline)
        return pipeline

    namespace["create"] = classmethod(create)
    bound = type(name, (Pipeline,), namespace)
    _BOUND_PIPELINES[problem.name] = bound
    return bound
