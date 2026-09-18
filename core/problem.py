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
    """

    operator: ForwardOperator
    g: torch.Tensor
    f_true: Optional[torch.Tensor]
    mode: DataMode

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

    >> load_measurement : callable(config) -> torch.Tensor
        REAL mode: read the acquisition from disk and apply any preprocessing, returning g.

    >> load_truth : callable(config) -> torch.Tensor, or None
        SYNTHETIC mode: read the known object f_true. None means the problem supports real
        data only, and asking for SYNTHETIC mode then raises a clear error.

    >> simulate : callable(operator, f_true) -> torch.Tensor
        SYNTHETIC mode: produce the measurement from the truth. Defaults to `operator.apply`,
        i.e. g = H f_true. Override for a problem whose simulation is not just the forward
        model (MA-TIRF, for instance, also preprocesses the simulated stack).

    >> validate : callable(config) -> list[str]
        Returns one human-readable message per missing or invalid setting, and an empty
        list when the config is ready. Never raises: the caller shows the whole list at
        once so the user fixes everything in one pass.

    >> save_image / load_image : callable
        How this problem's images are written to and read from disk (TIF for a 3D stack,
        PNG for a 2D image).

    >> image_extension : str
        Extension used when saving results, e.g. "TIF" or "png".

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
    build_operator: Optional[Callable[[dict], ForwardOperator]] = None
    load_truth: Optional[Callable[[dict], torch.Tensor]] = None
    simulate: Optional[Callable[[ForwardOperator, torch.Tensor], torch.Tensor]] = None
    description: str = ""

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

    def prepare(self, config: dict) -> PreparedProblem:
        """
        Turn a configuration into everything a run needs: the operator, g, and f_true.

        The two modes differ only in where the measurement comes from:
            > REAL       g is loaded from disk; f_true is None
            > SYNTHETIC  f_true is loaded, then g is simulated from it

        Call `validate(config)` first — `prepare` assumes the config is already usable and
        will fail with the underlying loader's error otherwise.
        """
        mode = DataMode.from_config(config)
        operator = self.make_operator(config)
        if mode is DataMode.REAL:
            return PreparedProblem(operator, self.load_measurement(config), None, mode)
        if not self.supports_synthetic:
            raise ValueError(
                f"Problem {self.name!r} does not support synthetic mode "
                f"(it declares no load_truth), but its config asks for it."
            )
        f_true = self.load_truth(config)
        simulate = self.simulate or (lambda op, f: op.apply(f))
        return PreparedProblem(operator, simulate(operator, f_true), f_true, mode)

    def __repr__(self) -> str:
        feats = ", ".join(sorted(str(f) for f in self.features)) or "none"
        synth = "real+synthetic" if self.supports_synthetic else "real only"
        return f"<InverseProblem {self.name!r} features={{{feats}}} {synth}>"
