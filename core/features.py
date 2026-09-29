"""
Catalogue of the properties that characterize an inverse problem.

In v1 these were bare strings ("3d", "anisotropic") scattered across four layers, with
nothing to tell a reader which features existed or what each one actually changed. Here
they are a single typed enum that carries its own documentation.

A feature is a FACT about the problem, never a preference. "This problem is 3D" is a
feature; "use 200 iterations" is a parameter. The distinction matters: features are what
the framework matches against to decide what applies.

Who reads them:
    > a ForwardOperator declares the features it induces      (operator.features)
    > an InverseProblem exposes them                          (problem.features)
    > a solver / regularizer / metric declares what it needs  (.requires)
    > a UI parameter declares what it needs                   ("requires" in its ui dict)

The rule is always the same: an item applies when its `requires` is a subset of the
problem's features. `supports()` below is that rule, written once.

Feature is str-backed, so Feature.THREE_D == "3d" is True. That is deliberate: v1 code
comparing against raw strings keeps working while the migration proceeds.
"""

from enum import Enum
from typing import Iterable


class Feature(str, Enum):
    """
    A property of an inverse problem that changes what the framework may apply to it.

    ----------
    > Members :
    ----------

    >> TWO_D / THREE_D
        Dimensionality of the reconstruction space. Gates the differential operators
        (gradient, divergence, laplacian, hessian), the viewers used by the GUI, and
        metrics that only make sense in one of the two (e.g. FSC is 3D).
        A problem declares exactly one of them.

    >> ANISOTROPIC
        Voxels are not cubic — the axial sampling step differs from the lateral one
        (MA-TIRF: tens of nm in z, hundreds in xy). Gates the `delta` parameter
        (delta = dz / dxy) and the regularizers that weight the axial derivative with it.
        Without this feature, a regularizer may assume isotropic voxels.

    >> SCALE_AMBIGUOUS
        The forward model determines f only up to a positive multiplicative constant, so
        f and alpha*f explain the measurement equally well. Any metric comparing to a
        ground truth must therefore fit the optimal scale first (see `optimal_scale`),
        otherwise PSNR/MSE measure the arbitrary scale rather than the reconstruction.
    """

    TWO_D = "2d"
    THREE_D = "3d"
    ANISOTROPIC = "anisotropic"
    SCALE_AMBIGUOUS = "scale_ambiguous"

    def __str__(self) -> str:
        return self.value

    @property
    def explanation(self) -> str:
        """One-line description, read from this class's docstring — used by `catalogue()`."""
        return _EXPLANATIONS[self]


_EXPLANATIONS = {
    Feature.TWO_D: "reconstruction space is 2D",
    Feature.THREE_D: "reconstruction space is 3D",
    Feature.ANISOTROPIC: "voxels are not cubic (axial step != lateral step)",
    Feature.SCALE_AMBIGUOUS: "f is determined only up to a positive scale factor",
}


## the canonical empty set, so callers can write `requires = NO_FEATURES` explicitly
NO_FEATURES: frozenset = frozenset()


def features(*items: Iterable) -> frozenset:
    """
    Build a feature set, accepting Features or their raw string values.

        features(Feature.THREE_D, Feature.ANISOTROPIC)
        features("3d", "anisotropic")               # equivalent (migration-friendly)

    Raises ValueError on an unknown string, which turns a silent typo — the classic
    failure mode of the v1 string sets — into an immediate, explicit error.
    """
    out = []
    for item in items:
        try:
            out.append(Feature(item))
        except ValueError:
            valid = ", ".join(f.value for f in Feature)
            raise ValueError(f"Unknown feature {item!r}. Valid features: {valid}") from None
    return frozenset(out)


def supports(required, available) -> bool:
    """
    The single matching rule of the framework: does `available` satisfy `required`?

    True when `required` is a subset of `available` — so an item requiring nothing applies
    everywhere, and an item requiring {THREE_D} applies only to 3D problems.

        supports(AdamSolver.requires, problem.features)
        supports(param.get("requires", NO_FEATURES), problem.features)
    """
    return frozenset(required or NO_FEATURES).issubset(frozenset(available or NO_FEATURES))


def catalogue() -> str:
    """Human-readable list of every feature — what `<problem> features` can print."""
    width = max(len(f.value) for f in Feature)
    return "\n".join(f"  {f.value:<{width}}  {f.explanation}" for f in Feature)
