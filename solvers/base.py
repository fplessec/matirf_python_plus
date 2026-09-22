"""
Solver — the base class of every reconstruction algorithm.

A solver answers one question: given an Objective, find the f that minimizes it. It never
sees the forward operator's internals, never loads data, never touches a config file, and
never knows which inverse problem it is serving.

--------------------------------------------------------------------------------------
What changed from v1, and why it matters
--------------------------------------------------------------------------------------

v1's `Algorithm.run(g, H, params)` took the raw operator tensor, so it had to know how to
use it. Every inverse problem therefore re-declared every algorithm just to plug in
`apply_forward` / `apply_adjoint`:

    matirf/algorithms/   6 subclasses + a forward-model mixin   ~330 lines
    deconv/algorithms/   2 subclasses + a forward-model mixin   ~140 lines

All of it plumbing. Here `solve(objective, f0, params)` receives everything it needs
through the Objective, so a solver is written once and works for every problem — present
and future. Those two packages disappear, and adding an inverse problem requires no
algorithm code at all.

The same inversion removes the layering violation v1 could not escape: an algorithm no
longer reaches into a problem (nor, transitively, into that problem's GUI), so the
circular imports that forced lazy-import workarounds throughout v1 cannot occur.

--------------------------------------------------------------------------------------
The two halves of this class
--------------------------------------------------------------------------------------

    the algorithm    `solve()` — the only thing a subclass writes
    the machinery    running it off the main thread, reporting progress, interrupting it,
                     and publishing snapshots for live preview — written once, here

Subclasses touch only the first. Nothing below `── execution ──` should need overriding.
"""

import random
import threading
import traceback
from abc import ABC, abstractmethod

import numpy as np
import torch

from core.features import NO_FEATURES, supports
from solvers.fidelities import NOISE_MODEL_NAMES


def ridge_weight(objective, value) -> float:
    """
    The ridge weight lambda_rr as a number; unset or "auto" means the operator's largest
    singular value s1 = sqrt(||H^T H||), from a power iteration — available for any operator.

    (H^T H + lambda I)^-1 H^T keeps the singular directions with s^2 >> lambda and cuts the
    others. On MA-TIRF only 2-3 directions per depth column are determined by the data
    (singular values 39, 6.0, 0.56, 0.03, ... for 13 angles over 50 planes): any lambda
    between s2 and s1 keeps the determined ones and cuts the noise-dominated rest; below s3
    the inverse amplifies noise, far above s1 (the old 1e4) it barely acts.
    """
    if value in (None, "", "auto", "None", "null"):
        like = objective.operator.adjoint(objective.g)
        return float(objective.operator.lipschitz(like)) ** 0.5
    return float(value)



def second_eigenvalue(objective) -> float:
    """
    s2^2, the second eigenvalue of H^T H, when the operator exposes its spectrum
    (`singular_values()`, e.g. MA-TIRF), else s1^2 = ||H^T H|| from a power iteration.

    It is the ridge weight a start (H^T H + lambda I)^-1 H^T g is compared to: s2^2 keeps the
    best determined direction fully and the second half-way, and scales with the operator's
    gain (on MA-TIRF s2^2 = 36, inside the [s2, s1] = [6, 39] range the MCMC study found good).
    """
    spectrum = getattr(objective.operator, "singular_values", None)
    if spectrum is not None:
        values = spectrum()
        if values.numel() > 1:
            return float(values[1]) ** 2
    like = objective.operator.adjoint(objective.g)
    return float(objective.operator.lipschitz(like))


def ridge_start(objective, params: dict) -> torch.Tensor:
    """
    The start 'init' asks for: 'ridge' (default) with lambda_rr, empty = s2^2, or 'adjoint'.
    Used by the v2 solvers, whose starts must have the right scale and shape.
    """
    operator, g = objective.operator, objective.g
    if params.get("init", "ridge") == "adjoint":
        return operator.adjoint(g).detach().clamp(min=0.0)
    weight = params.get("lambda_rr")
    weight = second_eigenvalue(objective) if weight in (None, "", "None", "null", "auto") \
        else float(weight)
    return operator.ridge_inverse(g, weight).detach().clamp(min=0.0)

class Solver(ABC):
    """
    Base class for reconstruction algorithms.

    ----------
    > Class attributes (declared by each solver) :
    ----------

    >> name : str
        Identifier shown in the UI and stored in the config, e.g. "ADAM".

    >> estimator_type : str
        "MAP" for solvers returning the most probable f (Adam, PPXA, ADMM, PnP), "MMSE" for
        those returning a posterior mean (MCMC). Purely descriptive; it tells the user what
        kind of answer they are getting.

    >> requires : frozenset[Feature]
        Problem features this solver needs. Empty means it works on any problem — which is
        the normal case now that solvers are problem-agnostic. `supported_by()` applies the
        framework's single matching rule.

    >> supported_noise_models : frozenset[str]
        The noise models (fidelity `name`s) this solver can actually minimize. The noise
        model is chosen with the MEASUREMENT, in the '[noise-model]' section, not with the
        algorithm: it describes the detector, like H describes the optics. What a solver
        declares is only whether it knows how to handle it. Adam differentiates any D, so
        it takes them all; a splitting solver whose data step is a least-squares solve
        (PPXA, ADMM, PnP, PnP-ADMM) takes only 'gaussian'. Choosing another is refused at
        validation with a readable message — never silently replaced by the Gaussian step,
        which is what v1's PPXA did.

    >> uses_regularization / uses_denoiser : bool
        Whether the solver consults the objective's regularization term, or a denoiser.
        Used by the UI to show only the parameters that actually affect this solver.

    >> ui_params : dict
        The solver's parameters, in the shared params-UI format. Pure data: types, defaults
        and labels only. It must contain no reference to the GUI layer — that coupling is
        exactly what produced v1's circular imports. Buttons that compute a value (an
        "Estimate" action) are attached by the GUI, from a capability of the operator.

    ----------
    > To implement :
    ----------

    >> solve(objective, f0, params) -> torch.Tensor
        The algorithm. Start from `f0`, minimize `objective`, return the result. Inside the
        iteration loop, call `self.report(...)` to log, `self.publish(f)` to feed the live
        preview, and check `self.interrupted` to honour a stop request.

    ----------
    > Example :
    ----------

        class GradientDescent(Solver):
            name = "GD"
            uses_regularization = True
            ui_params = {"max_iter": {...}, "step": {...}}

            def solve(self, objective, f0, params):
                f, step = f0, params["step"]
                for i in range(params["max_iter"]):
                    if self.interrupted:
                        return f
                    f = (f - step * objective.grad(f)).clamp_(min=0)
                    self.publish(f)
                return f
    """

    name: str = ""
    estimator_type: str = "MAP"
    requires: frozenset = NO_FEATURES
    supported_noise_models: frozenset = NOISE_MODEL_NAMES
    uses_regularization: bool = False
    uses_denoiser: bool = False
    ui_params: dict = {}

    # ── applicability and documentation ──────────────────────────────────────

    @classmethod
    def supported_by(cls, problem_features) -> bool:
        """True when a problem's features satisfy this solver's `requires`."""
        return supports(cls.requires, problem_features)

    @classmethod
    def get_ui_params(cls, problem_features=None) -> dict:
        """
        Every parameter the user may set for this solver, filtered by the problem.

        Two sources are merged, so a solver declares only what is specific to it:
            > the prior and its weight (reg, lambda_reg, delta, rho), when `uses_regularization`
            > this solver's own `ui_params`, which override the shared ones on a key clash
        See solvers/objective_params.py.

        A parameter may declare `"requires": {...}` and is dropped when the problem lacks
        those features. That is how `delta` (the anisotropy ratio) disappears on an
        isotropic problem without any solver writing a conditional.
        """
        from solvers.objective_params import REGULARIZATION_UI_PARAMS

        available = problem_features if problem_features is not None else cls.requires
        merged = {}
        if cls.uses_regularization:
            merged.update(REGULARIZATION_UI_PARAMS)
        merged.update(cls.ui_params)
        return {
            key: param for key, param in merged.items()
            if supports(param.get("requires", NO_FEATURES), available)
        }

    @classmethod
    def description(cls) -> str:
        """Assembled from the docstrings along the inheritance chain, as in v1."""
        parts = [f"[{cls.name}] ({cls.estimator_type} estimator)"]
        for klass in reversed(cls.__mro__):
            if klass.__doc__ and klass not in (object, ABC, Solver):
                parts.append(klass.__doc__.strip())
        parts.append(f"Noise models: {', '.join(sorted(cls.supported_noise_models))}")
        parts.append(f"Uses regularization: {cls.uses_regularization}")
        parts.append(f"Uses denoiser: {cls.uses_denoiser}")
        return "\n\n".join(parts)

    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._latest = None
        ## copying the iterate costs a full-size clone per iteration; the pipeline turns it off
        ## when nobody watches (headless runs, or the `live_preview` setting unticked):
        self.publishing = True
        self.start_point = None
        ## reporting hooks, wired by the pipeline; harmless no-ops when running standalone,
        ## which is what lets a solver be used directly from a script or a notebook:
        self.on_message = lambda msg: None
        self.on_finished = lambda f: None
        self.on_error = lambda err: None

    # ── the algorithm ────────────────────────────────────────────────────────

    @abstractmethod
    def solve(self, objective, f0: torch.Tensor, params: dict) -> torch.Tensor:
        """Minimize `objective` starting from `f0`. Returns the reconstructed tensor."""

    def initial_guess(self, objective, params: dict) -> torch.Tensor:
        """
        Where the iteration starts. Chosen by a parameter, not by a subclass.

            "adjoint"  f0 = H^T g        cheap back-projection (the default)
            "ridge"    f0 = (H^T H + lam I)^-1 H^T g    a warm start much closer to the
                       solution, at the cost of one linear solve

        In v1 the ridge start was a MA-TIRF-only subclass override, because only MA-TIRF's
        mixin knew how to invert its operator. `ForwardOperator.ridge_inverse` now provides
        it for every problem, so the choice becomes an ordinary parameter and the override
        disappears.
        """
        strategy = params.get("init", "adjoint")
        operator, g = objective.operator, objective.g
        if strategy == "ridge":
            return operator.ridge_inverse(g, ridge_weight(objective, params.get("lambda_rr", 1e4))).detach()
        if strategy == "adjoint":
            return operator.adjoint(g).detach().clone()
        raise ValueError(f"Unknown init strategy {strategy!r}; expected 'adjoint' or 'ridge'.")

    # ── execution: threading, reporting, interruption ────────────────────────
    # Written once. Subclasses use `report` / `publish` / `interrupted` and nothing else.

    def report(self, message: str) -> None:
        """Log a line: to the GUI message panel, or stdout in CLI mode."""
        self.on_message(message)

    def publish(self, f: torch.Tensor) -> None:
        """
        Publish a snapshot of the current iterate for the live preview.

        Stored rather than signalled: the UI polls it on a timer. Emitting a signal per
        iteration would flood the Qt event loop and freeze the interface — a lesson kept
        from v1.

        A no-op when `publishing` is False, so solvers call it unconditionally.
        """
        if self.publishing:
            self._latest = f.detach().clone()

    @property
    def latest(self):
        """The most recent published iterate, or None before the first `publish`."""
        return self._latest

    @property
    def interrupted(self) -> bool:
        """True once a stop has been requested. Check it inside the iteration loop."""
        return self._stop_event.is_set()

    def run_in_background(self, objective, params: dict) -> None:
        """Run `solve` in a daemon thread, reporting completion or failure through the hooks."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._guarded_solve, args=(objective, params), daemon=True)
        self._thread.start()

    def _guarded_solve(self, objective, params: dict) -> None:
        try:
            ## kept, so what the solver returns can be compared with where it started
            self.start_point = self.initial_guess(objective, params)
            f = self.solve(objective, self.start_point, params)
            if not self.interrupted:
                self.on_finished(f)
        except Exception as e:
            traceback.print_exc()
            self.on_error(f"{type(e).__name__}: {e}")

    def stop(self, timeout: float = 3.0) -> None:
        """Request interruption and wait for the thread to finish its current iteration."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self.report("Interruption requested.")
            self._thread.join(timeout=timeout)

    @staticmethod
    def fix_randomness(seed: int = 123) -> None:
        """Seed every RNG, so a run with the same config is reproducible."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} {self.estimator_type}>"
