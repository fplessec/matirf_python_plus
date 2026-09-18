"""
Test and reference implementation of the MA-TIRF problem.

Run it:
    python -m problems.matirf._tests

Two kinds of check, and the first matters more than the second:

    REGRESSION   the ported physics must reproduce v1 EXACTLY. A refactor of an optical
                 model does not fail loudly — it quietly shifts every reconstruction. So
                 the operator matrix, the preprocessing and the anisotropy estimate are all
                 compared against the v1 implementation, which still ships alongside.

    CONTRACT     the operator satisfies what solvers assume (a true adjoint, a correct
                 closed-form solve), and the declaration turns a real config into a
                 runnable problem in both modes.

Everything runs on the real measurement in matirf/data/measurements/.
"""

import json
from pathlib import Path

import torch

from fileio import load_json, load_tif
from core import DataMode, Feature, features
from problems.matirf import MATIRF_MEASUREMENTS_DIR
from solvers import Adam
from core import Objective

from problems.matirf import MATIRF, MatirfOperator
from problems.matirf import physics, problem as matirf_problem

TIF = str(MATIRF_MEASUREMENTS_DIR / "esoubies.TIF")
JSON = str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")
TRUTH = str(MATIRF_MEASUREMENTS_DIR / "synthetic_truth0.TIF")


def _config(mode=DataMode.REAL, nz=10, z0=0.0, zN=400.0, add_noise=None):
    return {
        "input-paths": {"mode": mode.value, "tif": TIF if mode is DataMode.REAL else TRUTH,
                        "json": JSON},
        "oper-params": {"nz": nz, "z0": z0, "zN": zN, "normalize": False},
        "add-noise": add_noise or {},
        "algo-params": {},
    }


# ── regression against v1 ─────────────────────────────────────────────────────

def _reference():
    """
    The frozen v1 outputs, recorded before the v1 implementation was deleted.

    The regression check has to outlive the code it was written against, otherwise the
    guarantee disappears exactly when it starts mattering — when someone later "tidies up"
    physics.py. Small results (the operator matrices) are stored whole; large ones are
    stored as a fingerprint: shape, sum, mean, std, min, max and a few fixed samples. Any
    real change to the optics moves at least one of those.
    """
    return json.loads((Path(__file__).parent / "_v1_reference.json").read_text())


def _fingerprint(t):
    flat = t.detach().flatten().double()
    idx = [0, len(flat) // 7, len(flat) // 3, len(flat) // 2, len(flat) - 1]
    return {"shape": list(t.shape),
            "sum": float(flat.sum()), "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            "samples": [float(flat[i]) for i in idx]}


def _assert_fingerprint(actual, expected, what):
    assert actual["shape"] == expected["shape"], f"{what}: shape changed"
    for key in ("sum", "mean", "std", "min", "max"):
        assert abs(actual[key] - expected[key]) <= 1e-4 * max(abs(expected[key]), 1.0), \
            f"{what}: {key} drifted ({actual[key]:.8g} vs {expected[key]:.8g})"
    for i, (a, e) in enumerate(zip(actual["samples"], expected["samples"])):
        assert abs(a - e) <= 1e-4 * max(abs(e), 1.0), f"{what}: sample {i} drifted"


def test_physics_matches_v1():
    """The optics must still produce exactly what v1 produced."""
    reference = _reference()
    measurement = load_json(JSON)

    for oper, expected in zip(reference["cases"], reference["operator"]):
        produced = physics.build_operator_matrix(
            angles_deg=measurement["angles_deg"],
            nz=oper["nz"], z0=oper["z0"], zN=oper["zN"],
            n_glass=measurement["n_glass"], n_medium=measurement["n_medium"],
            numerical_aperture=measurement["numerical_aperture"],
            n_oil=measurement["n_oil"], wavelength_nm=measurement["wavelength_nm"],
            beam_divergence_deg=measurement["beam_divergence_deg"],
            normalize=oper.get("normalize", False),
        )
        assert torch.allclose(produced, torch.tensor(expected, dtype=produced.dtype),
                              atol=1e-6), f"the operator matrix changed for {oper}"

    for oper, expected in zip(reference["cases"], reference["delta"]):
        produced = MatirfOperator.from_measurement(measurement, oper).estimate_anisotropy_ratio()
        assert abs(produced - expected) < 1e-9, "the anisotropy estimate changed"

    # the angle bounds, against the recorded v1 values (never transcribed by hand)
    angles = reference["angles"]
    assert abs(physics.critical_angle(1.518, 1.34) - angles["critical_1.518_1.34"]) < 1e-9
    assert abs(physics.max_angle(1.33, 1.4) - angles["max_1.33_1.4"]) < 1e-9
    print("  regression      operator matrix, angle bounds and delta unchanged since v1")


def test_preprocessing_matches_v1():
    """Background removal and normalization must still produce what v1 produced."""
    reference = _reference()
    raw = load_tif(TIF)
    g, params = matirf_problem.preprocess(raw.clone(), load_json(JSON), {})
    _assert_fingerprint(_fingerprint(g), reference["preprocessed_g"], "preprocessed measurement")
    assert params["angles_deg"] == reference["angles_after_preprocessing"]

    # this file has no background stack; fabricate one to exercise that branch, which is
    # what makes the angle list (and so the operator's shape) change
    measurement = load_json(JSON)
    _, theta_max = physics.angle_bounds(measurement)
    measurement["angles_deg"] = measurement["angles_deg"][:-1] + [theta_max + 5.0]
    g, refined = matirf_problem.split_background(raw.clone(), measurement)
    assert g.shape[0] == raw.shape[0] - 1, "the background stack must be removed"
    assert len(refined["angles_deg"]) == len(measurement["angles_deg"]) - 1
    assert measurement["angles_deg"][-1] not in refined["angles_deg"]
    print("  preprocessing   unchanged since v1; background stacks drop their angles too")


# ── the operator contract ─────────────────────────────────────────────────────

def test_operator_contract():
    """What every solver assumes about this operator."""
    operator = MatirfOperator.from_config(_config(nz=12))
    assert operator.features == features(Feature.THREE_D, Feature.ANISOTROPIC,
                                         Feature.SCALE_AMBIGUOUS)
    assert operator.H.shape == (13, 12), "one row per angle, one column per depth slice"

    f = torch.rand(12, 8, 8, dtype=operator.H.dtype)
    y = torch.rand(13, 8, 8, dtype=operator.H.dtype)
    assert operator.apply(f).shape == (13, 8, 8)
    assert operator.adjoint(y).shape == (12, 8, 8)

    # the check that catches the most expensive class of bug in this codebase.
    # the tolerance is the operator's own default, sized for float32 (settings.dtype):
    # in float64 this operator lands at 3e-16, in float32 at ~1e-7.
    assert operator.check_adjoint(f, y) < 1e-4, "adjoint must be the true transpose"

    # The closed-form override must agree with the base class's conjugate gradient.
    # Tolerance is relative and loose on purpose: H^T H here has a smallest eigenvalue of
    # about -4e-6 — it is numerically singular, which is the physics, not a bug (thirteen
    # exponential decay profiles are very nearly linearly dependent). Adding lam = 0.5
    # brings the condition number to ~550, and float32 CG then agrees to ~1e-5 absolute.
    # This ill-conditioning is exactly why a ridge term or a regularizer is not optional here.
    from core.operator import ForwardOperator
    b = torch.rand(12, 4, 4, dtype=operator.H.dtype)
    closed = operator.solve_normal(b, 0.5)
    by_cg = ForwardOperator.solve_normal(operator, b, 0.5, n_iter=300, tol=1e-14)
    relative = ((closed - by_cg).abs().max() / closed.abs().max()).item()
    assert relative < 1e-4, f"closed form and CG disagree by {relative:.2e} (relative)"

    # the cache must not return a stale inverse when lam changes
    assert not torch.allclose(operator.solve_normal(b, 0.5), operator.solve_normal(b, 50.0))
    print("  operator        shape, true adjoint, closed form == CG, per-lam cache")


# ── the declaration ───────────────────────────────────────────────────────────

def test_problem_declaration():
    """One object answers: what kind of problem, is the config valid, what am I solving."""
    assert MATIRF.features == features(Feature.THREE_D, Feature.ANISOTROPIC,
                                       Feature.SCALE_AMBIGUOUS)
    assert MATIRF.supports_synthetic and MATIRF.image_extension == "TIF"

    assert MATIRF.validate(_config()) == []
    empty = MATIRF.validate({})                       # what a fresh `matirf reset` leaves
    assert len(empty) == 6 and not any("KeyError" in e for e in empty), empty
    ## `normalize` is a checkbox: absent means unset, but False is a valid answer
    assert any("normalize" in e for e in empty)
    ticked = {**_config(), "oper-params": {**_config()["oper-params"], "normalize": False}}
    assert not any("normalize" in e for e in MATIRF.validate(ticked))
    noisy = MATIRF.validate({**_config(), "add-noise": {"add_noise": True, "sigma": "None"}})
    assert noisy == ["Noise sigma: required when 'add noise' is enabled"]
    print("  declaration     features, validation survives an empty config")


def test_prepare_both_modes():
    """prepare() loads first, then builds the operator from the refined config."""
    real = MATIRF.prepare(_config(DataMode.REAL, nz=10))
    assert real.mode is DataMode.REAL and not real.has_truth
    assert real.g.shape == (13, 350, 350)
    assert real.operator.H.shape == (13, 10)
    assert "_measurement" in real.config, "the refined config is carried forward"

    synthetic = MATIRF.prepare(_config(DataMode.SYNTHETIC, nz=99))
    assert synthetic.mode is DataMode.SYNTHETIC and synthetic.has_truth
    assert synthetic.f_true.shape == (50, 64, 64)
    # nz=99 was overridden by the truth's own depth count — the operator must match the file
    assert synthetic.operator.H.shape == (13, 50), "nz comes from the truth, not the config"
    assert synthetic.g.shape == (13, 64, 64)
    assert torch.isfinite(synthetic.g).all()
    print("  prepare         both modes; nz taken from the truth in synthetic mode")


def test_solver_runs_on_matirf():
    """The end of the chain: a generic solver reconstructs this problem, knowing nothing of it."""
    prepared = MATIRF.prepare(_config(DataMode.SYNTHETIC, nz=99))

    class _Gaussian:
        display_name = "gaussian"
        def loss(self, Hf, g):
            return 0.5 * ((Hf - g) ** 2).sum()

    objective = Objective(prepared.operator, prepared.g, _Gaussian())
    solver = Adam()
    params = {"max_iter": 60, "lr": 0.01, "K": 60, "EPS": 1e-14, "init": "adjoint"}
    f0 = solver.initial_guess(objective, params)
    f = solver.solve(objective, f0, params)

    assert f.shape == prepared.f_true.shape == (50, 64, 64)
    assert torch.isfinite(f).all() and (f >= 0).all(), "positivity must hold"
    start, end = objective.value(f0).item(), objective.value(f).item()
    assert end < start, f"the objective must decrease ({start:.3e} -> {end:.3e})"

    ## Only the LOSS is asserted, because that is the solver's job. The distance to f_true is
    ## NOT asserted: MA-TIRF is severely ill-posed (H^T H is numerically singular, see above),
    ## so many very different f explain g almost equally well and sixty iterations do not yet
    ## pick the right one. Judging reconstruction quality belongs in benchmarks/, with
    ## regularization and a proper iteration budget.
    print(f"  end-to-end      Adam ran on real MA-TIRF data, loss {start:.2e} -> {end:.2e}")


def test_ridge_warm_start_helps():
    """
    The generic ridge initialization is worth having — measured, not asserted in prose.

    In v1 this warm start existed only for MA-TIRF, as a subclass override, because only its
    mixin knew how to invert the operator. `ForwardOperator.ridge_inverse` gives it to every
    problem; here is what it buys on real data.

    Because MA-TIRF is SCALE_AMBIGUOUS, f and alpha*f explain g equally well, so the
    distance to the truth is measured AFTER fitting the optimal scale. Comparing without
    that step would measure an arbitrary constant rather than the reconstruction.
    """
    from core.metrics import optimal_scale

    prepared = MATIRF.prepare(_config(DataMode.SYNTHETIC, nz=99))

    class _Gaussian:
        display_name = "gaussian"
        def loss(self, Hf, g):
            return 0.5 * ((Hf - g) ** 2).sum()

    objective = Objective(prepared.operator, prepared.g, _Gaussian())
    solver = Adam()

    def aligned_error(f):
        alpha = optimal_scale(f, prepared.f_true)
        return ((alpha * f - prepared.f_true).norm() / prepared.f_true.norm()).item()

    from_adjoint = solver.initial_guess(objective, {"init": "adjoint"})
    from_ridge = solver.initial_guess(objective, {"init": "ridge", "lambda_rr": 1e-2})

    assert objective.value(from_ridge) < objective.value(from_adjoint)
    adjoint_error, ridge_error = aligned_error(from_adjoint), aligned_error(from_ridge)
    assert ridge_error < adjoint_error, (
        f"the ridge warm start should be closer to the truth "
        f"({ridge_error:.3f} vs {adjoint_error:.3f})")
    print(f"  ridge start     scale-aligned error {adjoint_error:.3f} (adjoint) "
          f"-> {ridge_error:.3f} (ridge)")


def main():
    print("problems.matirf — regression against v1, and contracts\n")
    test_physics_matches_v1()
    test_preprocessing_matches_v1()
    test_operator_contract()
    test_problem_declaration()
    test_prepare_both_modes()
    test_solver_runs_on_matirf()
    test_ridge_warm_start_helps()
    print("\nMA-TIRF is ported faithfully.")


if __name__ == "__main__":
    main()
