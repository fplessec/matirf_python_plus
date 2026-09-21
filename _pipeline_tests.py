"""
Test and reference implementation of the Pipeline.

Run it:
    python -m _pipeline_tests

This is the first point where the whole v2 stack runs together: a problem declaration, a
generic solver, an assembled objective, metrics against a known truth, and a saved result —
with no GUI anywhere. If this file passes, the framework works; the interface is only a way
to fill in the config.

It doubles as the shortest complete example of driving a reconstruction from a script,
which is what a benchmark or a batch job does.
"""

import tempfile
import time
from pathlib import Path

import torch

from core.enums import PipelineState
from core import DataMode
from problems.deconv import DECONV_MEASUREMENTS_DIR
from problems.matirf import MATIRF_MEASUREMENTS_DIR
from pipeline import Pipeline, build_objective, resolve_noise_model, validate_noise_model
from problems.deconv import DECONV
from problems.matirf import MATIRF


def _deconv_config(algorithm="ADAM", **algo_params):
    return {
        "algorithm": algorithm,
        "input-paths": {"mode": DataMode.SYNTHETIC.value,
                        "png": str(DECONV_MEASUREMENTS_DIR / "img_001.png"),
                        "json": str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")},
        "add-noise": {},
        "algo-params": {"max_iter": 30, "lr": 0.02, "K": 30, "EPS": 1e-14, **algo_params},
    }


def _matirf_config(algorithm="ADAM", **algo_params):
    return {
        "algorithm": algorithm,
        "input-paths": {"mode": DataMode.SYNTHETIC.value,
                        "tif": str(MATIRF_MEASUREMENTS_DIR / "synthetic_truth0.TIF"),
                        "json": str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")},
        "oper-params": {"nz": 50, "z0": 0.0, "zN": 400.0, "normalize": False},
        "add-noise": {},
        "algo-params": {"max_iter": 20, "lr": 0.01, "K": 20, "EPS": 1e-14, **algo_params},
    }


def _run_to_completion(problem, config, timeout=120.0):
    """Start a pipeline and block until it finishes — what a script or a benchmark does."""
    pipeline = Pipeline.create(problem, config)
    errors = []
    pipeline.on_error = errors.append
    pipeline.start()
    deadline = time.time() + timeout
    while pipeline.is_running and time.time() < deadline:
        time.sleep(0.02)
    assert not errors, f"the run reported an error: {errors}"
    Pipeline.remove(pipeline)
    return pipeline


# ── validation ────────────────────────────────────────────────────────────────

def test_validation():
    """Every problem is reported at once, and the solver is checked too."""
    config = _deconv_config(algorithm="None")
    pipeline = Pipeline.create(DECONV, config)
    messages, errors = [], []
    pipeline.on_message, pipeline.on_error = messages.append, errors.append
    pipeline.start()
    Pipeline.remove(pipeline)

    assert errors, "an unselected algorithm must stop the run"
    assert any("Algorithm: not selected" in m for m in messages)
    assert pipeline.state is PipelineState.IDLE, "a rejected config must not change state"

    # an algorithm name that is not a registered solver is caught with the available list
    pipeline = Pipeline.create(DECONV, _deconv_config(algorithm="NOPE"))
    messages = []
    pipeline.on_message, pipeline.on_error = messages.append, lambda e: None
    pipeline.start()
    Pipeline.remove(pipeline)
    assert any("unknown solver" in m and "ADAM" in m for m in messages)

    # a config emptied by `reset` produces a readable list, never a KeyError
    pipeline = Pipeline.create(DECONV, {"algorithm": "None", "algo-params": {}})
    messages = []
    pipeline.on_message, pipeline.on_error = messages.append, lambda e: None
    pipeline.start()
    Pipeline.remove(pipeline)
    assert any("PNG" in m for m in messages) and any("JSON" in m for m in messages)
    print("  validation      all issues at once, unknown solver named, empty config survives")


# ── assembling the objective ──────────────────────────────────────────────────

def test_build_objective():
    """The config becomes an Objective here — the step v1 asked every algorithm to repeat."""
    prepared = DECONV.prepare(_deconv_config()["input-paths"] and _deconv_config())

    plain = build_objective(prepared, {})
    assert plain.data_fidelity.display_name == "L2 (Gaussian noise)", "gaussian is the default"
    assert not plain.is_regularized

    regularized = build_objective(prepared, {"reg": "L1 norm of the gradient",
                                             "lambda_reg": 0.2})
    assert regularized.is_regularized and regularized.lambda_reg == 0.2
    assert regularized.data_weight == 0.8, "the v1 blend convention is applied"
    assert "L1 norm of the gradient" in regularized.describe()

    # a solver with a hardcoded quadratic data step is given no prior at all
    none_given = build_objective(prepared, {"reg": "L1 norm", "lambda_reg": 0.5},
                                 uses_regularization=False)
    assert not none_given.is_regularized

    # delta is asked of the operator when the user never set it — MA-TIRF can estimate it
    matirf_prepared = MATIRF.prepare(_matirf_config())
    estimated = build_objective(matirf_prepared, {}).diff_ops.delta
    assert abs(estimated - matirf_prepared.operator.estimate_anisotropy_ratio()) < 1e-9
    assert build_objective(prepared, {}).diff_ops.delta == 1.0, "isotropic by default"
    print("  objective       defaults, blend convention, prior withheld, delta estimated")


# ── the noise model ───────────────────────────────────────────────────────────

def test_noise_model():
    """
    The noise model is chosen with the measurement and checked against the solver.

    Resolved three ways — oracle (the simulation's own values), estimated (from g), manual —
    and refused, with a readable message, when the solver cannot minimize it.
    """
    from solvers import SOLVERS
    noisy = {"poisson_noise": True, "photons": 400.0, "gaussian_noise": True, "sigma": 0.01}
    config = {**_deconv_config(), "add-noise": noisy,
              "noise-model": {"fidelity": "Poisson-Gaussian", "parameters": "oracle"}}
    prepared = DECONV.prepare(config)

    # oracle: exactly what corrupted the simulation
    fidelity, message = resolve_noise_model(prepared, config)
    assert (fidelity.a, fidelity.b) == (1 / 400.0, 0.01 ** 2)
    assert "oracle" in message and "Poisson-Gaussian" in message

    # estimated from g alone: the same noise, recovered within 30 %
    estimated = {**config, "noise-model": {"fidelity": "Poisson-Gaussian",
                                           "parameters": "estimated"}}
    fidelity, _ = resolve_noise_model(prepared, estimated)
    assert abs(fidelity.a * 400 - 1) < 0.3 and abs(fidelity.b / 1e-4 - 1) < 0.3, \
        (fidelity.a, fidelity.b)

    # manual, and a Gaussian model uses b only
    manual = {**config, "noise-model": {"fidelity": "L2 (Gaussian noise)",
                                        "parameters": "manual", "b": 0.004}}
    fidelity, _ = resolve_noise_model(prepared, manual)
    assert fidelity.name == "gaussian" and fidelity.b == 0.004

    # an old config kept the fidelity among the algorithm parameters: still honoured
    legacy = {**_deconv_config(data_fidelity="KL divergence (Poisson noise)")}
    assert resolve_noise_model(prepared, legacy)[0].name == "poisson"

    # refused: a solver whose data step is least squares, given a Poisson model...
    poisson = {**config, "noise-model": {"fidelity": "KL divergence (Poisson noise)"}}
    errors = validate_noise_model(poisson, SOLVERS["ADMM"])
    assert errors and "ADMM handles only gaussian" in errors[0]
    assert validate_noise_model(poisson, SOLVERS["ADAM"]) == []
    # ...oracle values on a real measurement, and manual values left empty
    real = {**config, "input-paths": {**config["input-paths"], "mode": DataMode.REAL.value}}
    assert any("synthetic mode" in e for e in validate_noise_model(real))
    empty = {**config, "noise-model": {"fidelity": "Poisson-Gaussian", "parameters": "manual"}}
    assert len(validate_noise_model(empty)) == 2, "both a and b are required"

    # the refusal stops the run before any computation, with the reason in the log
    pipeline = Pipeline.create(DECONV, {**poisson, "algorithm": "PPXA"})
    messages = []
    pipeline.on_message, pipeline.on_error = messages.append, lambda e: None
    pipeline.start()
    Pipeline.remove(pipeline)
    assert any("PPXA handles only gaussian" in m for m in messages)
    assert pipeline.state is PipelineState.IDLE
    print("  noise model     oracle exact, estimated within 30 %, manual; ADMM+Poisson refused")


# ── full runs ─────────────────────────────────────────────────────────────────

def test_deconv_run():
    """A complete deconvolution run, evaluated against the known truth."""
    pipeline = _run_to_completion(DECONV, _deconv_config())
    result = pipeline.result

    assert pipeline.state is PipelineState.COMPLETED
    assert result.is_complete() and result.has_synthetic_truth()
    assert result.f.shape == result.f_true.shape
    assert result.metrics and "MSE" in result.metrics and "PSNR" in result.metrics
    assert "FSC" not in result.metrics, "FSC is 3D-only and must be skipped on a 2D problem"
    assert result.diff is not None and result.alpha is not None
    assert "Solver: ADAM" in result.messages and "[state]" in result.messages
    print(f"  deconv run      completed, PSNR={result.metrics['PSNR']:.2f} dB, "
          f"{len(result.metrics)} metrics")


def test_matirf_run():
    """The same pipeline, the same solver, a completely different problem."""
    pipeline = _run_to_completion(MATIRF, _matirf_config())
    result = pipeline.result

    assert pipeline.state is PipelineState.COMPLETED
    assert result.f.shape == (50, 64, 64)
    assert result.metrics and "FSC" in result.metrics, "FSC applies to a 3D problem"
    assert "Scale_alpha" in result.metrics, "a scale-ambiguous problem reports its scale"
    assert result.alpha is not None and result.diff.shape == result.f.shape
    print(f"  matirf run      completed, {len(result.metrics)} metrics including FSC")


def test_every_solver_through_the_pipeline():
    """All six solvers drive a real problem through the pipeline, selected only by name."""
    budgets = {
        "ADAM": {"max_iter": 20, "lr": 0.02, "K": 20},
        "PPXA": {"max_iter": 15, "lambda_relax": 1.0, "gamma": 1.0, "K": 15},
        "ADMM": {"iter": 8, "mu": 0.05, "threshold_ratio": 0.0},
        "PNP": {"iter": 4, "sigma": 5.0, "denoiser": "None"},
        "ADMM-PnP": {"iter": 8, "rho": 0.05, "sigma": 5.0, "denoiser": "None"},
        "MCMC": {"max_iter": 20, "beta": 1e-4, "sigma": 0.005, "K": 20, "lambda_rr": 0.05},
    }
    for name, params in budgets.items():
        config = _deconv_config(algorithm=name)
        config["algo-params"] = params
        pipeline = _run_to_completion(DECONV, config)
        assert pipeline.state is PipelineState.COMPLETED, name
        assert torch.isfinite(pipeline.result.f).all(), name
        assert pipeline.result.metrics, name
    print("  all solvers     six solvers ran through the pipeline, chosen by name alone")


# ── persistence and interruption ──────────────────────────────────────────────

def test_save_and_load():
    """A saved run is self-contained: it can be reopened without the session that made it."""
    pipeline = _run_to_completion(DECONV, _deconv_config())
    with tempfile.TemporaryDirectory() as directory:
        pipeline.save_results(directory)
        written = {path.name for path in Path(directory).iterdir()}
        assert written == {"f.png", "config.toml", "messages.txt", "f_true.png", "metrics.json"}

        reopened = Pipeline.create(DECONV, _deconv_config())
        reopened.load_results(directory)
        Pipeline.remove(reopened)
        assert reopened.state is PipelineState.COMPLETED
        assert reopened.result.f.shape == pipeline.result.f.shape
        assert reopened.result.metrics.keys() == pipeline.result.metrics.keys()
        assert "Solver: ADAM" in reopened.result.messages, "the log survives the round trip"
    print("  save / load     five files written, reopened without the original session")


def test_interruption():
    """A long run stops on request, and the registry lets several be stopped at once."""
    config = _deconv_config()
    ## K is small so a preview snapshot exists quickly: a solver publishes every K
    ## iterations, so a large K leaves the live preview blank for a long time.
    config["algo-params"] = {"max_iter": 5_000_000, "lr": 1e-4, "K": 5}
    ## a headless run publishes nothing (no copy per iteration); a display window switches
    ## live_preview on, as the `live_preview` setting asks — both are run side by side here
    quiet = Pipeline.create(DECONV, config)
    watched = Pipeline.create(DECONV, config)
    watched.live_preview = True
    for pipeline in (quiet, watched):
        pipeline.on_error = lambda e: None
        pipeline.start()
    time.sleep(0.6)

    for pipeline in (quiet, watched):
        assert pipeline.is_running and pipeline.state is PipelineState.COMPUTING
    assert quiet.latest_preview is None, "nobody watches: the iterate must not be copied"
    assert watched.latest_preview is not None, "a live-preview snapshot must be available"

    assert len(Pipeline.get_all()) == 2
    Pipeline.stop_all()
    for pipeline in (quiet, watched):
        assert pipeline.state is PipelineState.INTERRUPTED and not pipeline.is_running
    assert Pipeline.get_all() == []
    print("  interruption    stopped mid-run, preview only when watched, registry cleared")


def main():
    print("pipeline — the whole v2 stack, without a GUI\n")
    test_validation()
    test_build_objective()
    test_noise_model()
    test_deconv_run()
    test_matirf_run()
    test_every_solver_through_the_pipeline()
    test_save_and_load()
    test_interruption()
    print("\nThe pipeline runs both problems end to end.")


if __name__ == "__main__":
    main()
