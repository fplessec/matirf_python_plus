"""
InverseProblem — the single declaration of one inverse problem.

In v1, answering "what is the MA-TIRF problem?" meant reading six files: the package
__init__ (features, paths, default config), the pipeline (seven class attributes), the
pipeline operations (build g/H, validation, metrics, preview), the physics module, an
algorithm mixin (the forward model), and the GUI. Nothing stated the problem in one place.

Here it is one object. It answers exactly three questions, and nothing else:

    1. What kind of problem is this?          -> features (read from the operator)
    2. Is this configuration usable?          -> validate(config)
    3. Given a config, what am I solving?     -> prepare(config) -> operator, g, f_true

Everything downstream — solvers, metrics, GUI — is built from those answers, which is why
adding an inverse problem needs no algorithm code and no GUI code.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import torch

from .operator import ForwardOperator


class DataMode(str, Enum):
    """
    Where the measurement comes from.

    >> REAL       g is an actual acquisition loaded from disk; there is no ground truth,
                  so the reconstruction can only be judged qualitatively.
    >> SYNTHETIC  a known truth f_true is loaded and the measurement is simulated as
                  g = H f_true (noise added afterwards by the pipeline). Because the exact
                  answer is known, quality metrics are meaningful — this is the mode used
                  to compare algorithms.

    String-backed with the same values as v1 ("real-data" / "synthetic-data"), so existing
    config.toml files keep working through the migration.
    """

    REAL = "real-data"
    SYNTHETIC = "synthetic-data"

    @classmethod
    def from_config(cls, config: dict) -> "DataMode":
        """Read the mode out of a config, defaulting to REAL when the key is absent."""
        raw = (config.get("input-paths", {}) or {}).get("mode", cls.REAL.value)
        return cls(raw)


@dataclass(frozen=True)
class PreparedProblem:
    """
    Everything a solver run needs, produced by `InverseProblem.prepare`.

    >> operator : ForwardOperator   the physics, configured for this run
    >> g        : torch.Tensor      the measurement to explain
    >> f_true   : torch.Tensor|None the ground truth — SYNTHETIC mode only, else None
    >> mode     : DataMode          which of the two branches produced this
    >> config   : dict              the config AFTER loading refined it (for MA-TIRF, its
                                    angle list no longer contains the background stacks).
                                    Downstream steps — metrics, saved metadata — must use
                                    this one, not the config originally passed in.
    """

    operator: ForwardOperator
    g: torch.Tensor
    f_true: Optional[torch.Tensor]
    mode: DataMode
    config: dict = field(default_factory=dict)

    @property
    def has_truth(self) -> bool:
        return self.f_true is not None


@dataclass(frozen=True)
class InverseProblem:
    """
    One inverse problem, declared entirely by the fields below.

    ----------
    > Fields :
    ----------

    >> name : str
        Identifier used by the CLI, the cache directory and saved metadata, e.g. "matirf".

    >> operator_class : type[ForwardOperator]
        The problem's forward operator class. It is the single source of truth for the
        problem's `features`, so they are never restated anywhere else.

    >> build_operator : callable(config) -> ForwardOperator
        Builds a configured operator instance. Defaults to `operator_class.from_config`,
        which is what a problem normally provides.

    >> load_measurement : callable(config) -> g, or (g, refined_config)
        REAL mode: read the acquisition from disk and apply any preprocessing.

        A loader may return a REFINED CONFIG alongside the data, and the operator is then
        built from that refined config rather than the original. This is not a convenience:
        for MA-TIRF, preprocessing detects background stacks and removes them, which changes
        the list of incident angles — and H is built from exactly that list. Loading and
        operator construction are genuinely ordered, so `prepare` loads first.

    >> load_truth : callable(config) -> f_true, or (f_true, refined_config)
        SYNTHETIC mode: read the known object f_true, with the same refinement contract
        (MA-TIRF takes the number of z slices from the truth's own shape). None means the
        problem supports real data only, and asking for SYNTHETIC mode raises a clear error.

    >> simulate : callable(operator, f_true, config) -> torch.Tensor
        SYNTHETIC mode: produce the measurement from the truth. Defaults to `operator.apply`,
        i.e. g = H f_true. Override when the simulation is more than the forward model —
        MA-TIRF also normalizes the simulated stack and adds the configured noise.

    >> validate : callable(config) -> list[str]
        Returns one human-readable message per missing or invalid setting, and an empty
        list when the config is ready. Never raises: the caller shows the whole list at
        once so the user fixes everything in one pass.

    >> save_image / load_image : callable
        How this problem's images are written to and read from disk (TIF for a 3D stack,
        PNG for a 2D image).

    >> image_extension : str
        Extension used when saving results, e.g. "TIF" or "png".

    >> raw_path_key : str
        Key under '[input-paths]' holding the measurement file, e.g. "tif" or "png". Used
        by `preview` to show the file as it is on disk, next to its preprocessed form.

    ----------
    > Example :
    ----------

        MATIRF = InverseProblem(
            name="matirf",
            operator_class=MatirfOperator,
            load_measurement=load_and_preprocess_stack,
            load_truth=load_tif,
            simulate=simulate_matirf_measurement,
            validate=validate_matirf_config,
            save_image=save_tif, load_image=load_tif, image_extension="TIF",
        )
    """

    name: str
    operator_class: type
    load_measurement: Callable[[dict], torch.Tensor]
    validate: Callable[[dict], list]
    save_image: Callable
    load_image: Callable
    image_extension: str
    raw_path_key: str = "tif"
    build_operator: Optional[Callable[[dict], ForwardOperator]] = None
    load_truth: Optional[Callable[[dict], torch.Tensor]] = None
    simulate: Optional[Callable[[ForwardOperator, torch.Tensor], torch.Tensor]] = None
    description: str = ""

    # ── where this problem keeps its files ───────────────────────────────────
    config_path: Optional[object] = None        # the cached config.toml
    default_config: Optional[dict] = None       # its shape when absent
    results_dir: Optional[object] = None        # where reconstructions are saved
    measurements_dir: Optional[object] = None   # where the file dialogs open

    # ── how its solvers start by default ─────────────────────────────────────
    # {solver name: {parameter: value}}, applied UNDER the user's '[algo-params]': a value
    # the user set always wins. It exists because the right default can be a property of
    # the problem, not of the algorithm — on MA-TIRF, Adam and PPXA start from the ridge
    # estimate (lambda_rr = 1e4), as v1 did; from the raw back-projection H^T g, which is
    # ~50x too large there, Adam ends in a meaningless, depth-shifted image.
    solver_defaults: dict = field(default_factory=dict)

    # ── the interface ────────────────────────────────────────────────────────
    # A `gui.spec.ProblemUI`, typed as a plain object on purpose: `core` must not import
    # `gui`. The problem CARRIES its interface declaration without depending on the code
    # that draws it, so the drawing can be replaced without touching a single problem.
    ui: Optional[object] = None

    def solver_params(self, solver_name: str, params: dict) -> dict:
        """The parameters a solver actually receives: this problem's defaults, then the user's."""
        return {**self.solver_defaults.get(solver_name, {}), **(params or {})}

    # ── what kind of problem this is ─────────────────────────────────────────

    @property
    def features(self) -> frozenset:
        """
        The problem's features, read from the operator class.

        Declared in exactly one place — the operator — because they are facts about the
        physics. Solvers, regularizers, metrics and UI parameters all match against this.
        """
        return self.operator_class.features

    @property
    def supports_synthetic(self) -> bool:
        return self.load_truth is not None

    # ── turning a config into a runnable problem ─────────────────────────────

    def make_operator(self, config: dict) -> ForwardOperator:
        """Build the configured operator, via `build_operator` or `operator_class.from_config`."""
        if self.build_operator is not None:
            return self.build_operator(config)
        from_config = getattr(self.operator_class, "from_config", None)
        if from_config is None:
            raise TypeError(
                f"{self.operator_class.__name__} has no from_config(config) classmethod, so "
                f"problem {self.name!r} must pass build_operator=... explicitly."
            )
        return from_config(config)

    @staticmethod
    def _load(loader, config):
        """
        Run a loader, accepting either `data` or `(data, refined_config)`.

        Letting a loader refine the config is what keeps the ordering correct: the data
        itself can determine part of the operator's configuration (see `load_measurement`).
        """
        loaded = loader(config)
        if isinstance(loaded, tuple):
            data, refined = loaded
            return data, refined
        return loaded, config

    def prepare(self, config: dict) -> PreparedProblem:
        """
        Turn a configuration into everything a run needs: the operator, g, and f_true.

        Order matters and is the same in both modes — LOAD FIRST, then build the operator
        from the (possibly refined) config, because loading can determine what the operator
        should be:

            > REAL       g is loaded and preprocessed; f_true is None
            > SYNTHETIC  f_true is loaded, then g is simulated from it

        Call `validate(config)` first — `prepare` assumes the config is already usable and
        will otherwise fail with the underlying loader's error.
        """
        mode = DataMode.from_config(config)

        if mode is DataMode.REAL:
            g, config = self._load(self.load_measurement, config)
            return PreparedProblem(self.make_operator(config), g, None, mode, config)

        if not self.supports_synthetic:
            raise ValueError(
                f"Problem {self.name!r} does not support synthetic mode "
                f"(it declares no load_truth), but its config asks for it."
            )
        f_true, config = self._load(self.load_truth, config)
        operator = self.make_operator(config)
        simulate = self.simulate or (lambda op, f, cfg: op.apply(f))
        return PreparedProblem(operator, simulate(operator, f_true, config), f_true, mode, config)

    def preview(self, config: dict) -> tuple:
        """
        The two images the GUI shows side by side before a run: input, and what the
        pipeline will actually feed the solver.

        It answers the question a user asks before committing to a reconstruction — "is my
        preprocessing doing what I think?" — and it answers it with the SAME code path the
        run will take, so a preview can never disagree with the run that follows.

            REAL       (the file as loaded, the measurement after preprocessing)
            SYNTHETIC  (the ground truth, the measurement simulated from it)

        May raise: an inconsistent file or a missing parameter surfaces here rather than at
        the start of a long run, which is the point.
        """
        mode = DataMode.from_config(config)
        if mode is DataMode.SYNTHETIC:
            prepared = self.prepare(config)
            return prepared.f_true, prepared.g
        raw = self.load_image(config["input-paths"][self.raw_path_key])
        preprocessed, _ = self._load(self.load_measurement, config)
        return raw, preprocessed

    def __repr__(self) -> str:
        feats = ", ".join(sorted(str(f) for f in self.features)) or "none"
        synth = "real+synthetic" if self.supports_synthetic else "real only"
        return f"<InverseProblem {self.name!r} features={{{feats}}} {synth}>"
