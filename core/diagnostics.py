"""
Is this reconstruction plausible? — detecting solutions that are mathematically fine but
physically meaningless.

A solver can converge, lower its loss and still return nonsense. Four failure shapes recur,
and each points at a parameter or an implementation problem rather than at the data:

    COLLAPSED     (3D) the object is crushed into the first planes, near the interface,
                  and the deeper planes are empty. H barely sees depth in some directions,
                  so a solver with too little regularization or a bad step size can put
                  everything where the operator is most sensitive.
    Z-INVARIANT   (3D) every plane shows the same 2D image: no depth has been recovered
                  at all. Typically a learning rate too small, or an initialization the
                  solver never left.
    MEASUREMENT   f ~ g: the solver stayed in the measurement space (when f and g have the
                  same shape — deconvolution) and never entered the solution space.
    TRIVIAL       f ~ the back-projection H^T g, the ridge estimate, or the solver's own
                  starting point: least-squares answers, correct but not reconstructions.

Every comparison is SCALE-INVARIANT — the distance between two images is the sine of the
angle between them — because MA-TIRF determines f only up to a positive factor.

When the truth is known (synthetic mode) it is used as the reference: a truth that is
itself shallow is not "collapsed", and a reconstruction merely no better than the ridge
estimate is flagged even if it looks different from it.

    diagnose(f, prepared, f0=None, lambda_rr=None) -> Diagnosis
        .findings   list of Finding(code, severity, message)
        .rejected   True when any finding is a "reject"
        .summary()  one line for a run log

The thresholds below were calibrated on reconstructions judged good by hand (the saved
gt0/gt2 results) and on constructed failures; see core/_tests.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from .features import Feature

REJECT, WARN = "reject", "warn"

## a plane holds "nothing" below this fraction of the most filled plane's energy
EMPTY_PLANE = 0.01
## COLLAPSED without a truth: this much energy in the top 10 % of the planes, with most
## planes empty
COLLAPSED_TOP_ENERGY = 0.8
COLLAPSED_EMPTY_FRACTION = 0.5
## COLLAPSED with a truth: the depth profiles differ by this much (1-Wasserstein distance,
## as a fraction of the depth range)
DEPTH_SHIFT_REJECT, DEPTH_SHIFT_WARN = 0.25, 0.10
## Z-INVARIANT: planes this correlated with their mean, over a nearly flat depth profile
Z_INVARIANT_CORRELATION = 0.97
Z_INVARIANT_FLATNESS = 0.5
## MEASUREMENT / TRIVIAL: closer than this (sine of the angle) to g, H^T g, ridge or f0
TRIVIAL_DISTANCE = 0.05


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str


@dataclass
class Diagnosis:
    findings: list = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return any(finding.severity == REJECT for finding in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "Realism check: plausible reconstruction."
        head = "REJECTED" if self.rejected else "warnings"
        return f"Realism check {head}: " + " | ".join(
            f"[{finding.severity}] {finding.message}" for finding in self.findings)


# ── scale-invariant comparisons ───────────────────────────────────────────────

def angle_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """sin of the angle between x and y: 0 when proportional, 1 when orthogonal."""
    x, y = x.detach().flatten().double(), y.detach().flatten().double()
    nx, ny = x.norm(), y.norm()
    if nx == 0 or ny == 0:
        return 1.0
    cosine = float((x @ y) / (nx * ny))
    return max(0.0, 1.0 - cosine * cosine) ** 0.5


def depth_profile(f: torch.Tensor) -> torch.Tensor:
    """Energy per plane (axis 0), normalized to sum to 1."""
    profile = f.detach().double().clamp(min=0).sum(dim=tuple(range(1, f.dim())))
    total = profile.sum()
    return profile / total if total > 0 else profile


def depth_shift(f: torch.Tensor, truth: torch.Tensor) -> float:
    """1-Wasserstein distance between two depth profiles, as a fraction of the depth range."""
    cumulative = torch.cumsum(depth_profile(f) - depth_profile(truth), dim=0)
    return float(cumulative.abs().sum()) / max(1, f.shape[0] - 1)


def plane_correlation(f: torch.Tensor) -> float:
    """Median correlation of the non-empty planes with their mean plane (shape, not level)."""
    planes = f.detach().double().clamp(min=0).flatten(1)
    energy = planes.sum(dim=1)
    filled = planes[energy > EMPTY_PLANE * energy.max()]
    if filled.shape[0] < 2:
        return 0.0
    centred = filled - filled.mean(dim=1, keepdim=True)
    mean_plane = centred.mean(dim=0)
    norms = centred.norm(dim=1) * mean_plane.norm()
    correlations = (centred @ mean_plane) / norms.clamp(min=1e-30)
    return float(correlations.median())


# ── the diagnosis ─────────────────────────────────────────────────────────────

def diagnose(f: torch.Tensor, prepared, f0: Optional[torch.Tensor] = None,
             lambda_rr: Optional[float] = None) -> Diagnosis:
    """Every sign that `f` is not a meaningful reconstruction of `prepared`."""
    diagnosis = Diagnosis()
    add = lambda code, severity, message: diagnosis.findings.append(
        Finding(code, severity, message))

    if not torch.isfinite(f).all():
        add("non-finite", REJECT, "f contains NaN or infinite values")
        return diagnosis
    if float(f.detach().abs().max()) <= 1e-12:
        add("empty", REJECT, "f is zero everywhere")
        return diagnosis

    truth = prepared.f_true
    operator, g = prepared.operator, prepared.g
    if Feature.THREE_D in getattr(operator, "features", ()) and f.dim() == 3 and f.shape[0] > 2:
        _check_depth(f, truth, add)

    back_projection = operator.adjoint(g)
    if f.shape == g.shape:
        ## "close to g" is relative to how much H changes an image: under a mild blur a good
        ## reconstruction is itself near g. H^T g is g blurred once more, so its distance
        ## to g measures that blur; f must stay well clear of g on that scale.
        limit = min(TRIVIAL_DISTANCE, 0.3 * angle_distance(back_projection, g))
        if angle_distance(f, g) < limit:
            add("measurement", REJECT,
                f"f ~ g (distance {angle_distance(f, g):.3f}): no reconstruction, the solver "
                f"stayed in the measurement space")

    references = {"the back-projection H^T g": back_projection}
    if lambda_rr is not None:
        references["the ridge estimate"] = operator.ridge_inverse(g, float(lambda_rr))
    if f0 is not None and all(angle_distance(f0, r) > 1e-6 for r in references.values()):
        references["the initial guess"] = f0
    for name, reference in references.items():
        distance = angle_distance(f, reference)
        if distance < TRIVIAL_DISTANCE:
            add("trivial", REJECT, f"f ~ {name} (distance {distance:.3f}): the solver did not "
                                   f"move away from a least-squares answer")
        elif truth is not None and angle_distance(f, truth) >= angle_distance(reference, truth):
            add("no-better", WARN, f"f is no closer to the truth than {name} "
                                   f"({angle_distance(f, truth):.3f} vs "
                                   f"{angle_distance(reference, truth):.3f})")
    return diagnosis


def _check_depth(f, truth, add):
    profile = depth_profile(f)
    n_planes = f.shape[0]
    top = max(1, round(0.1 * n_planes))
    empty = float((profile < EMPTY_PLANE * profile.max()).double().mean())
    top_energy = float(profile[:top].sum())

    if truth is not None and truth.shape == f.shape:
        shift = depth_shift(f, truth)
        truth_top = float(depth_profile(truth)[:top].sum())
        if shift > DEPTH_SHIFT_REJECT or (top_energy > COLLAPSED_TOP_ENERGY
                                          and truth_top < 0.5 * top_energy):
            add("collapsed", REJECT,
                f"depth profile far from the truth's (shift {shift:.2f} of the depth range, "
                f"{top_energy:.0%} of the energy in the top {top} planes vs {truth_top:.0%})")
        elif shift > DEPTH_SHIFT_WARN:
            add("depth-shift", WARN, f"depth profile shifted by {shift:.2f} of the depth range")
    elif top_energy > COLLAPSED_TOP_ENERGY and empty > COLLAPSED_EMPTY_FRACTION:
        add("collapsed", REJECT,
            f"{top_energy:.0%} of the energy in the top {top} of {n_planes} planes and "
            f"{empty:.0%} of the planes empty: crushed against the interface")

    correlation = plane_correlation(f)
    flatness = float(profile.min() / profile.max()) if profile.max() > 0 else 0.0
    truth_correlation = plane_correlation(truth) if truth is not None and truth.shape == f.shape \
        else 0.0
    if (correlation > Z_INVARIANT_CORRELATION and flatness > Z_INVARIANT_FLATNESS
            and truth_correlation < Z_INVARIANT_CORRELATION):
        add("z-invariant", REJECT,
            f"every plane shows the same image (correlation {correlation:.3f}, depth profile "
            f"flat to {flatness:.0%}): no depth recovered")
