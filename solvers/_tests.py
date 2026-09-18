"""
Test and reference implementation of the `solvers` layer.

Run it:
    python -m solvers._tests

The claim this layer makes is strong and worth testing directly: *one* implementation of an
algorithm serves *every* inverse problem. So the central test runs the same Adam instance
against two operators with nothing in common —

    _MatrixOperator   a dense matrix         (how MA-TIRF's forward model works)
    _BlurOperator     a circular convolution (how deconvolution's forward model works)

— and checks it recovers the truth in both cases. In v1 this required two subclasses in two
packages; here the solver is not even aware that the physics changed.

The operators are tiny and closed-form, so a failure here means a broken contract, not
numerical bad luck.
"""

import contextlib
import math
import os
import time

import torch

from core import Feature, features, ForwardOperator, Objective
from solvers import SOLVERS, available_for, Adam

DTYPE = torch.float64


# ── two structurally different physics ────────────────────────────────────────

class _MatrixOperator(ForwardOperator):
    """H is an explicit matrix — measurement space may differ in size from reconstruction."""

    name = "matrix"
    features = features(Feature.THREE_D, Feature.ANISOTROPIC)

    def __init__(self, matrix):
        self.H = matrix

    def apply(self, f):
        return self.H @ f

    def adjoint(self, y):
        return self.H.T @ y


class _BlurOperator(ForwardOperator):
    """H is a circular convolution, applied in Fourier — H is never formed explicitly."""

    name = "blur"
    features = features(Feature.TWO_D)

    def __init__(self, kernel, n):
        padded = torch.zeros(n, dtype=DTYPE)
        padded[:kernel.numel()] = kernel
        self.spectrum = torch.fft.fft(torch.roll(padded, -(kernel.numel() // 2)))

    def apply(self, f):
        return torch.fft.ifft(torch.fft.fft(f) * self.spectrum).real

    def adjoint(self, y):
        return torch.fft.ifft(torch.fft.fft(y) * self.spectrum.conj()).real


class _Gaussian:
    display_name = "gaussian"

    def loss(self, Hf, g):
        return 0.5 * ((Hf - g) ** 2).sum()


class _L2:
    display_name = "l2"

    def loss(self, f, diff_ops):
        return 0.5 * (f ** 2).sum()

    def prox(self, f, weight, diff_ops):
        return f / (1.0 + weight)


def _random_matrix_operator(m: int, n: int, seed: int = 0) -> _MatrixOperator:
    """
    A random (m, n) operator, scaled so that H^T H has eigenvalues around 1.

    The scaling matters and is not a test trick: without it H^T g — the default starting
    point — lands an order of magnitude away from f_true, and a gradient method needs
    thousands of extra iterations just to travel the distance. Real operators are
    normalized for exactly this reason (MA-TIRF does it in `normalize_operator`).

    Every test seeds explicitly, so results never depend on the order tests run in.
    """
    torch.manual_seed(seed)
    return _MatrixOperator(torch.randn(m, n, dtype=DTYPE) / math.sqrt(m))


# ── tests ─────────────────────────────────────────────────────────────────────

def test_registry():
    """The registry answers which solvers a given problem can run."""
    assert "ADAM" in SOLVERS and SOLVERS["ADAM"] is Adam
    # Adam requires nothing, so it is available to every problem — the normal case now
    assert "ADAM" in available_for(features(Feature.TWO_D))
    assert "ADAM" in available_for(features(Feature.THREE_D, Feature.ANISOTROPIC))
    assert Adam.supported_by(set())
    assert "MAP" in Adam.description() and "ADAM" in Adam.description()
    print("  registry        lookup, availability by features, description")


def test_ui_params():
    """A solver declares only what is specific to it; the objective parameters are merged in."""
    own = set(Adam.ui_params)
    assert {"max_iter", "lr", "K", "EPS"} <= own
    assert "lambda_reg" not in own, "objective parameters are not repeated in the solver"

    # merged for a solver that uses regularization:
    isotropic = Adam.get_ui_params(features(Feature.TWO_D))
    assert {"max_iter", "data_fidelity", "reg", "lambda_reg"} <= set(isotropic)

    # `delta` requires ANISOTROPIC: present on MA-TIRF-like problems, absent on 2D ones,
    # with no conditional anywhere in the solver
    assert "delta" not in isotropic
    anisotropic = Adam.get_ui_params(features(Feature.THREE_D, Feature.ANISOTROPIC))
    assert "delta" in anisotropic

    # a solver that ignores regularization gets none of the objective parameters
    class _Plain(Adam):
        uses_regularization = False
    assert "lambda_reg" not in _Plain.get_ui_params(features(Feature.TWO_D))
    print("  ui params       objective params merged, delta gated by ANISOTROPIC")


def test_initial_guess():
    """Where the iteration starts is a parameter now, not a problem-specific subclass."""
    op = _random_matrix_operator(10, 6, seed=1)
    objective = Objective(op, torch.randn(10, dtype=DTYPE), _Gaussian())
    solver = Adam()

    adjoint_start = solver.initial_guess(objective, {})
    assert torch.allclose(adjoint_start, op.adjoint(objective.g)), "default is f0 = H^t g"

    # the ridge warm start was a MA-TIRF-only override in v1; it is generic now
    ridge_start = solver.initial_guess(objective, {"init": "ridge", "lambda_rr": 1.0})
    expected = op.ridge_inverse(objective.g, 1.0)
    assert torch.allclose(ridge_start, expected, atol=1e-8)

    try:
        solver.initial_guess(objective, {"init": "nonsense"})
        raise AssertionError("an unknown init strategy must raise")
    except ValueError as e:
        assert "Unknown init strategy" in str(e)
    print("  initial guess   adjoint default, generic ridge warm start, bad value rejected")


def _recover(operator, f_true, params, lambda_reg=0.0):
    """Simulate g = H f_true, then let Adam reconstruct f from it."""
    objective = Objective(operator, operator.apply(f_true), _Gaussian(),
                          _L2() if lambda_reg else None, lambda_reg=lambda_reg)
    solver = Adam()
    f0 = solver.initial_guess(objective, params)
    f = solver.solve(objective, f0, params)
    return (f - f_true).norm() / f_true.norm()


def test_same_solver_two_physics():
    """The point of the whole layer: one Adam, two unrelated forward models."""
    params = {"max_iter": 3000, "lr": 0.05, "K": 500, "EPS": 1e-14}

    # physics 1: a dense matrix, more measurements than unknowns
    f_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=DTYPE)
    matrix_op = _random_matrix_operator(14, 6, seed=0)
    assert matrix_op.check_adjoint(f_true, matrix_op.apply(f_true)) < 1e-10
    matrix_error = _recover(matrix_op, f_true, params)
    assert matrix_error < 1e-6, f"matrix problem not recovered ({matrix_error:.2e})"

    # physics 2: a circular convolution, computed in Fourier — nothing in common with above
    n = 32
    signal = torch.zeros(n, dtype=DTYPE)
    signal[8:14] = 1.0
    signal[20] = 2.0
    blur_op = _BlurOperator(torch.tensor([0.25, 0.5, 0.25], dtype=DTYPE), n)
    assert blur_op.check_adjoint(signal, signal) < 1e-10
    blur_error = _recover(blur_op, signal, {**params, "lr": 0.02, "max_iter": 6000})
    assert blur_error < 1e-6, f"blur problem not recovered ({blur_error:.2e})"

    print(f"  agnosticism     SAME Adam solved a dense matrix ({matrix_error:.1e}) "
          f"and a convolution ({blur_error:.1e})")


def test_regularization_and_reporting():
    """The solver reads the objective's regularization without knowing what it is."""
    f_true = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    op = _random_matrix_operator(9, 4, seed=2)
    objective = Objective(op, op.apply(f_true), _Gaussian(), _L2(), lambda_reg=0.05)

    messages = []
    solver = Adam()
    solver.on_message = messages.append
    f = solver.solve(objective, solver.initial_guess(objective, {}),
                     {"max_iter": 400, "lr": 0.05, "K": 100, "EPS": 1e-14})

    assert (f >= 0).all(), "the positivity projection must hold on the returned tensor"
    assert any("gaussian" in m and "l2" in m for m in messages), \
        "the run log states what is being minimized"
    assert any("iter" in m for m in messages)
    assert solver.latest is not None, "publish() fed the live preview"
    print("  reporting       positivity, objective logged, live-preview snapshot")


def test_threading_and_interruption():
    """Running off the main thread and honouring a stop request."""
    op = _random_matrix_operator(40, 30, seed=3)
    objective = Objective(op, torch.randn(40, dtype=DTYPE), _Gaussian())

    done, failed = [], []
    solver = Adam()
    solver.on_finished = done.append
    solver.on_error = failed.append
    solver.run_in_background(objective, {"max_iter": 2_000_000, "lr": 1e-3, "K": 50})

    time.sleep(0.3)                     # let it iterate
    assert solver.latest is not None, "a snapshot should exist while running"
    solver.stop()
    assert solver.interrupted
    assert not failed, f"interruption must not be reported as an error: {failed}"
    assert not done, "an interrupted run does not report completion"

    # a failing solve surfaces through on_error rather than killing the thread silently
    class _Broken(Adam):
        def solve(self, objective, f0, params):
            raise RuntimeError("boom")
    broken = _Broken()
    broken.on_error = failed.append
    ## the solver prints the traceback to stderr on purpose (it is what makes a real crash
    ## diagnosable); swallow it here so this expected failure does not read as a test failure
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        broken.run_in_background(objective, {})
        time.sleep(0.2)
    assert failed and "boom" in failed[0]
    print("  threading       background run, clean interruption, errors reported")


def main():
    print("solvers — contract tests\n")
    test_registry()
    test_ui_params()
    test_initial_guess()
    test_same_solver_two_physics()
    test_regularization_and_reporting()
    test_threading_and_interruption()
    print("\nAll solver contracts hold.")


if __name__ == "__main__":
    main()
