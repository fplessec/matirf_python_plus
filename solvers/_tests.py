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
from solvers import (SOLVERS, available_for, Adam, Ppxa, Admm, AdmmV2, Pnp, PnpV2, PnpAdmm,
                     Mcmc, McmcV2)

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
    def quadratic_scale(self, n_pixels):   # 1/2 ||r||^2: a quadratic of weight 1
        return 1.0


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


def _step_signal(n: int) -> torch.Tensor:
    """A piecewise-constant 1D signal: a plateau and a spike, easy to judge by eye."""
    signal = torch.zeros(n, dtype=DTYPE)
    signal[8:14] = 1.0
    signal[20] = 2.0
    return signal


# ── tests ─────────────────────────────────────────────────────────────────────

def test_registry():
    """The registry answers which solvers a given problem can run."""
    ## the "v2" solvers are proposed alternatives, kept separate until their owner decides
    assert set(SOLVERS) == {"ADAM", "PPXA", "ADMM", "ADMMv2", "PNP", "PNPv2", "ADMM-PnP",
                            "MCMC", "MCMCv2"}
    assert SOLVERS["ADAM"] is Adam and SOLVERS["MCMC"] is Mcmc and SOLVERS["MCMCv2"] is McmcV2

    # no solver requires any feature any more, so every problem gets all of them — that is the
    # whole promise of the layer, and it is checked rather than asserted in prose:
    for feats in (features(Feature.TWO_D),
                  features(Feature.THREE_D, Feature.ANISOTROPIC, Feature.SCALE_AMBIGUOUS)):
        assert len(available_for(feats)) == 9, f"every solver must serve {feats}"

    assert Mcmc.estimator_type == McmcV2.estimator_type == "MMSE", "the posterior means"
    assert all(SOLVERS[n].estimator_type == "MAP" for n in SOLVERS if not n.startswith("MCMC"))
    assert "MAP" in Adam.description() and "ADAM" in Adam.description()
    print("  registry        nine solvers (the v2 ones are proposals), all available to every problem")


def test_ui_params():
    """A solver declares only what is specific to it; the objective parameters are merged in."""
    own = set(Adam.ui_params)
    assert {"max_iter", "lr", "K", "EPS"} <= own
    assert "lambda_reg" not in own, "objective parameters are not repeated in the solver"

    # merged for a solver that uses regularization:
    isotropic = Adam.get_ui_params(features(Feature.TWO_D))
    assert {"max_iter", "reg", "lambda_reg"} <= set(isotropic)

    # `delta` requires ANISOTROPIC: present on MA-TIRF-like problems, absent on 2D ones,
    # with no conditional anywhere in the solver
    assert "delta" not in isotropic
    anisotropic = Adam.get_ui_params(features(Feature.THREE_D, Feature.ANISOTROPIC))
    assert "delta" in anisotropic

    # the prior is offered only to the solvers that consult it
    for solver in (Adam, Ppxa):
        assert "lambda_reg" in solver.get_ui_params(features(Feature.TWO_D)), solver.name
    for solver in (Mcmc, McmcV2, Admm, AdmmV2, Pnp, PnpV2, PnpAdmm):
        assert "lambda_reg" not in solver.get_ui_params(features(Feature.TWO_D)), solver.name

    # the noise model is NOT an algorithm parameter any more: it belongs to the measurement
    # ('[noise-model]'), and each solver only declares which models it can minimize
    for solver in SOLVERS.values():
        assert "data_fidelity" not in solver.get_ui_params(features(Feature.TWO_D)), solver.name
    assert Adam.supported_noise_models == {"gaussian", "poisson", "poisson-gaussian"}
    assert Mcmc.supported_noise_models == McmcV2.supported_noise_models == Adam.supported_noise_models
    for solver in (Ppxa, Admm, AdmmV2, Pnp, PnpV2, PnpAdmm):
        assert solver.supported_noise_models == {"gaussian"}, solver.name
    print("  ui params       prior gated by solver, delta by ANISOTROPIC, noise models declared")


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


def test_solve_normal():
    """The primitive that made every splitting solver problem-agnostic."""
    op = _random_matrix_operator(12, 5, seed=4)
    b = torch.randn(5, dtype=DTYPE)
    lam = 0.6
    expected = torch.linalg.solve(op.H.T @ op.H + lam * torch.eye(5, dtype=DTYPE), b)
    assert torch.allclose(op.solve_normal(b, lam, n_iter=200, tol=1e-14), expected, atol=1e-8)

    # ridge_inverse is solve_normal applied to H^t y, so overriding one accelerates both
    y = torch.randn(12, dtype=DTYPE)
    assert torch.allclose(op.ridge_inverse(y, lam, n_iter=200, tol=1e-14),
                          op.solve_normal(op.adjoint(y), lam, n_iter=200, tol=1e-14))

    # an override is picked up by every solver without touching any of them
    class _Direct(_MatrixOperator):
        def solve_normal(self, b, lam=0.0, **kwargs):
            n = self.H.shape[1]
            return torch.linalg.solve(self.H.T @ self.H + lam * torch.eye(n, dtype=DTYPE), b)
    direct = _Direct(op.H)
    assert torch.allclose(direct.solve_normal(b, lam), expected, atol=1e-12)
    assert torch.allclose(direct.ridge_inverse(y, lam),
                          op.ridge_inverse(y, lam, n_iter=200, tol=1e-14), atol=1e-8)
    print("  solve_normal    matches the closed form, ridge_inverse built on it, override works")


def test_every_solver_on_both_physics():
    """
    The central claim, checked exhaustively: every solver runs on both physics.

    Each is given a modest budget and only has to get meaningfully closer to the truth than
    the back-projection it starts from. This is a contract test, not a benchmark — the real
    quality comparison belongs in benchmarks/, on real data.
    """
    budgets = {
        "ADAM": {"max_iter": 400, "lr": 0.05, "K": 200, "EPS": 1e-14},
        "PPXA": {"max_iter": 300, "lambda_relax": 1.0, "gamma": 0.5, "K": 150, "EPS": 1e-14},
        "ADMM": {"iter": 30, "mu": 0.1, "threshold_ratio": 0.0},
        "ADMMv2": {"iter": 100, "mu": 1.0, "kappa": 0.001},
        "PNP": {"iter": 8, "sigma": 5.0, "denoiser": "None", "kai_zhang": True},
        "PNPv2": {"iter": 6, "sigma": 5.0, "denoiser": "None"},
        "ADMM-PnP": {"iter": 20, "rho": 0.1, "sigma": 5.0, "denoiser": "None"},
        "MCMC": {"max_iter": 60, "beta": 1e-3, "sigma": 0.01, "K": 30, "lambda_rr": 1e-3},
        "MCMCv2": {"max_iter": 60, "sigma": 0.03, "K": 30},
    }

    problems = {
        "matrix": (_random_matrix_operator(14, 6, seed=0),
                   torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=DTYPE)),
        "blur": (_BlurOperator(torch.tensor([0.25, 0.5, 0.25], dtype=DTYPE), 32),
                 _step_signal(32)),
    }

    for physics, (operator, f_true) in problems.items():
        g = operator.apply(f_true)
        baseline = (operator.adjoint(g) - f_true).norm() / f_true.norm()
        for name, solver_class in SOLVERS.items():
            torch.manual_seed(7)
            objective = Objective(operator, g, _Gaussian())
            solver = solver_class()
            f0 = solver.initial_guess(objective, budgets[name])
            f = solver.solve(objective, f0, budgets[name])
            error = ((f - f_true).norm() / f_true.norm()).item()
            assert torch.isfinite(f).all(), f"{name} on {physics} produced non-finite values"
            assert error < baseline, (
                f"{name} on {physics}: error {error:.3e} is no better than the "
                f"back-projection it started from ({baseline:.3e})")
    print(f"  all solvers     {len(SOLVERS)} solvers x two physics: every run improved on H^t g")


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
    signal = _step_signal(32)
    blur_op = _BlurOperator(torch.tensor([0.25, 0.5, 0.25], dtype=DTYPE), 32)
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

    # nobody watching (headless run, or the live_preview setting off): no copy at all
    quiet = Adam()
    quiet.publishing = False
    quiet.solve(objective, quiet.initial_guess(objective, {}), {"max_iter": 20, "lr": 0.05, "K": 10})
    assert quiet.latest is None, "publish() must not copy the iterate when publishing is off"
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


# ── the Bayesian formulation: D is a scaled likelihood, lambda a pure prior weight ──

def test_fidelities_are_scaled_likelihoods():
    """
    Every fidelity is the per-pixel negative log-likelihood of its noise, so:
        D(g, g) = 0                  the constant is chosen that way
        D(H f_true, g) ~ 1/2         at the truth, for EVERY model with its right (a, b)
    The second is what makes lambda_reg mean the same thing with any noise model.
    """
    from core import noise
    from solvers.fidelities import GaussianFidelity, PoissonFidelity, PoissonGaussianFidelity

    clean = 0.05 + 0.95 * torch.rand(200_000, dtype=DTYPE)
    cases = [
        (GaussianFidelity, {"gaussian_noise": True, "sigma": 0.03}),
        (PoissonFidelity, {"poisson_noise": True, "photons": 300}),
        (PoissonGaussianFidelity, {"poisson_noise": True, "photons": 300,
                                   "gaussian_noise": True, "sigma": 0.03}),
    ]
    for cls, config in cases:
        model = noise.noise_model(config)
        g = noise.add_noise_to_measurement(clean, config)
        D = cls.from_noise(model.a, model.b)
        assert float(D.loss(g, g)) < 1e-12, f"{cls.__name__}: D(g, g) must be 0"
        at_truth = float(D.loss(clean, g))
        assert abs(at_truth - 0.5) < 0.05, f"{cls.__name__}: D at the truth = {at_truth:.3f}"

    # b = 1 is v1's 1/2 mean squared error, bit for bit
    a, b = torch.rand(50, dtype=DTYPE), torch.rand(50, dtype=DTYPE)
    assert torch.equal(GaussianFidelity().loss(a, b), 0.5 * torch.nn.functional.mse_loss(a, b))
    # the floor: a noiseless measurement does not make D infinite
    assert GaussianFidelity(b=0.0).b > 0 and PoissonFidelity(a=0.0).a > 0

    # nothing depends on the number of pixels: every D is a mean, so an image and the same
    # image repeated give the same value
    residual, data = torch.rand(40, 40, dtype=DTYPE), torch.rand(40, 40, dtype=DTYPE)
    for cls in (GaussianFidelity, PoissonFidelity, PoissonGaussianFidelity):
        D = cls.from_noise(0.01, 0.001)
        twice = D.loss(torch.cat([residual, residual]), torch.cat([data, data]))
        assert torch.allclose(D.loss(residual, data), twice), cls.__name__
    print("  fidelities      D(g,g) = 0, D(truth) = 1/2 for all three models, a mean (size-free)")


def test_adam_and_ppxa_minimize_the_same_objective():
    """
    A gradient solver and a proximal solver must agree on what L is.

    Adam differentiates L = (1 - lambda) D + lambda R directly; PPXA rebuilds it from a
    least-squares data step and the prior's prox. Before the scale convention (D and R both
    per-pixel means, prox of the MEAN), the two disagreed by the ratio of the image sizes
    and converged to different images. Now their minima coincide.
    """
    from solvers.differential_operators import DifferentialOperators
    from solvers.fidelities import GaussianFidelity
    from solvers.regularizers import TikhonovRegularization

    n = 32
    fy = torch.fft.fftfreq(n, dtype=DTYPE)
    kernel_hat = torch.exp(-2 * math.pi ** 2 * 1.5 ** 2 * (fy[:, None] ** 2 + fy[None, :] ** 2))

    class _Blur2D(ForwardOperator):
        name = "blur2d"
        features = features(Feature.TWO_D)
        def apply(self, f):
            return torch.fft.ifft2(torch.fft.fft2(f) * kernel_hat).real
        def adjoint(self, y):
            return self.apply(y)

    torch.manual_seed(3)
    f_true = torch.zeros(n, n, dtype=DTYPE)
    f_true[8:20, 6:14] = 1.0
    f_true[18:28, 16:26] = 0.6
    operator = _Blur2D()
    sigma = 0.03
    g = (operator.apply(f_true) + sigma * torch.randn(n, n, dtype=DTYPE)).clamp(min=0)

    objective = Objective(operator, g, GaussianFidelity(b=sigma ** 2), TikhonovRegularization(n_iter=60),
                          lambda_reg=0.02, diff_ops=DifferentialOperators())
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        adam = Adam()
        f_adam = adam.solve(objective, adam.initial_guess(objective, {}),
                            {"max_iter": 8000, "lr": 0.01, "K": 8000, "EPS": 0.0})
        ppxa = Ppxa()
        f_ppxa = ppxa.solve(objective, ppxa.initial_guess(objective, {}),
                            {"max_iter": 3000, "lambda_relax": 1.5, "gamma": 10.0,
                             "K": 3000, "EPS": 0.0})
    L_adam, L_ppxa = float(objective.value(f_adam)), float(objective.value(f_ppxa))
    gap = abs(L_adam - L_ppxa) / L_adam
    distance = float((f_adam - f_ppxa).norm() / f_adam.norm())
    ## a flat valley (lambda is small): L agrees much more tightly than the images do
    assert gap < 1e-3 and distance < 0.1, (
        f"Adam and PPXA disagree: L = {L_adam:.5f} vs {L_ppxa:.5f}, images {distance:.2%} apart")
    print(f"  same objective  Adam and PPXA reach the same minimum (L gap {gap:.1e}, "
          f"images {distance:.1e} apart)")


def test_mcmc_is_robust_to_its_temperature():
    """
    MCMCv2's claim, checked: beta is calibrated, so the relative temperature no longer decides
    whether the chain works, and lambda_rr defaults to s1. And MCMC itself has lost the
    meaningless `proposal_method` switch.
    """
    from solvers.base import ridge_weight
    op = _random_matrix_operator(40, 30, seed=11)
    f_true = torch.rand(30, dtype=DTYPE)
    objective = Objective(op, op.apply(f_true) + 0.01 * torch.randn(40, dtype=DTYPE), _Gaussian())
    assert abs(ridge_weight(objective, None) - float(torch.linalg.svdvals(op.H)[0])) < 1e-3 * \
        float(torch.linalg.svdvals(op.H)[0]), "automatic lambda_rr = the largest singular value"

    rates = []
    for temperature in (0.3, 1.0, 10.0):
        mcmc, log = McmcV2(), []
        mcmc.on_message = log.append
        mcmc.solve(objective, op.adjoint(objective.g),
                   {"max_iter": 100, "sigma": 0.03, "temperature": temperature, "K": 100})
        line = next(l for l in log if l.startswith("Accepted"))
        rates.append(float(line.split("(")[1].split("%")[0]))
    assert min(rates) > 50, f"a calibrated chain must move at any sensible temperature: {rates}"
    assert "proposal_method" not in Mcmc.ui_params and "proposal_method" not in McmcV2.ui_params
    print(f"  mcmcv2          acceptance {min(rates):.0f}-{max(rates):.0f}% for temperatures "
          f"0.3-10; lambda_rr auto = s1; no proposal switch")


def test_admm_v2_parameters_mean_what_they_say():
    """
    ADMMv2's two claims, checked against ADMM on a sparse toy problem: kappa = 1 empties the
    image exactly (tau = max(H^T g)), and mu changes only the speed, not the solution —
    while ADMM's mu changes the sparsity, and diverges well above the 1.618 bound.
    """
    torch.manual_seed(0)
    op = _random_matrix_operator(40, 60, seed=4)
    f_true = torch.zeros(60, dtype=DTYPE)
    f_true[[3, 17, 40]] = torch.tensor([1.0, 0.5, 0.8], dtype=DTYPE)
    objective = Objective(op, op.apply(f_true) + 0.01 * torch.randn(40, dtype=DTYPE), _Gaussian())

    assert float(AdmmV2().solve(objective, None, {"kappa": 1.0, "iter": 500}).norm()) == 0.0
    slow = AdmmV2().solve(objective, None, {"kappa": 0.05, "mu": 0.3, "iter": 5000})
    fast = AdmmV2().solve(objective, None, {"kappa": 0.05, "mu": 30.0, "iter": 5000})
    assert float((slow - fast).norm() / slow.norm()) < 1e-4, "mu must not change the solution"

    start = op.adjoint(objective.g)
    sparse = [int((Admm().solve(objective, start, {"iter": 2000, "mu": mu,
                                                    "threshold_ratio": 0.3}) > 1e-8).sum())
              for mu in (0.3, 1.5)]
    assert sparse[0] != sparse[1], "ADMM: mu changes the solution (the reason for ADMMv2)"
    diverged = Admm().solve(objective, start, {"iter": 2000, "mu": 2.5, "threshold_ratio": 0.3})
    assert not torch.isfinite(diverged).all(), "ADMM: mu = 2.5 is beyond its dual-step bound"
    print(f"  admm v2         kappa = 1 empties; mu speed-only (ADMM: {sparse[0]} vs {sparse[1]} "
          f"nonzeros for mu 0.3 / 1.5, NaN at 2.5)")


def test_pnp_v2_is_scale_free():
    """
    PNPv2's central claim: its denoising does not depend on the image's intensity scale.

    The same 2D problem with H's gain multiplied by c has a solution divided by c. With the
    data weight scaled consistently (lambda_kz x c^2, the eigenvalues of H^T H), PNPv2 with
    an intensity-based denoiser (Wiener) returns exactly the reconstruction divided by c;
    PnP, which assumes f in [0, 1], does not.
    """
    torch.manual_seed(1)
    n = 32
    fy = torch.fft.fftfreq(n, dtype=DTYPE)
    kernel = torch.exp(-2 * math.pi ** 2 * 1.2 ** 2 * (fy[:, None] ** 2 + fy[None, :] ** 2))

    class _Blur(ForwardOperator):
        name, features = "blur", features(Feature.TWO_D)
        def __init__(self, gain):
            self.gain = gain
        def apply(self, f):
            return self.gain * torch.fft.ifft2(torch.fft.fft2(f) * kernel).real
        def adjoint(self, y):
            return self.apply(y)

    f_true = torch.zeros(n, n, dtype=DTYPE)
    f_true[8:20, 6:14], f_true[18:28, 16:26] = 1.0, 0.6
    g = (_Blur(1.0).apply(f_true) + 0.02 * torch.randn(n, n, dtype=DTYPE)).clamp(min=0)

    def run(solver, gain, extra):
        objective = Objective(_Blur(gain), g, _Gaussian())
        params = {"iter": 8, "sigma": 20.0, "denoiser": "Wiener", "init": "ridge",
                  "lambda_rr": 0.1 * gain ** 2, **extra}
        return solver.solve(objective, solver.initial_guess(objective, params), params)

    c = 50.0
    v2 = [run(PnpV2(), gain, {"lambda_kz": 0.05 * gain ** 2}) for gain in (1.0, c)]
    v1 = [run(Pnp(), gain, {"lambda_kz": 0.05 * gain ** 2}) for gain in (1.0, c)]
    gap_v2 = float((c * v2[1] - v2[0]).norm() / v2[0].norm())
    gap_v1 = float((c * v1[1] - v1[0]).norm() / v1[0].norm())
    assert gap_v2 < 1e-4, f"PNPv2 must be scale-free (gap {gap_v2:.1e})"
    assert gap_v1 > 1e-2, f"PnP is expected to depend on the scale (gap {gap_v1:.1e})"
    print(f"  pnp v2          scale-free: gain x{c:g} gives the same image / {c:g} "
          f"(gap {gap_v2:.0e}; PnP: {gap_v1:.2f})")


def main():
    print("solvers — contract tests\n")
    test_registry()
    test_ui_params()
    test_initial_guess()
    test_solve_normal()
    test_every_solver_on_both_physics()
    test_same_solver_two_physics()
    test_regularization_and_reporting()
    test_threading_and_interruption()
    test_fidelities_are_scaled_likelihoods()
    test_adam_and_ppxa_minimize_the_same_objective()
    test_mcmc_is_robust_to_its_temperature()
    test_admm_v2_parameters_mean_what_they_say()
    test_pnp_v2_is_scale_free()
    print("\nAll solver contracts hold.")


if __name__ == "__main__":
    main()
