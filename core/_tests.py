"""
Test and reference implementation of the `core` layer.

Run it:
    python -m core._tests

`core` defines contracts, not implementations, so the honest way to test it is to build
the smallest possible inverse problem that satisfies those contracts and check that the
four objects compose. That is what this file is — and it doubles as the shortest complete
example of how a real problem is declared.

The toy problem is deliberately trivial: H is a small dense matrix, so every result can be
checked against a closed form computed independently with torch.linalg. When a test here
fails, the contract is broken — not the numerics.

Contents:
    _ToyOperator     a ForwardOperator over a small matrix (the "physics")
    _Gaussian        a data fidelity, D(Hf, g) = 0.5 ||Hf - g||^2
    _L2              a regularization with a closed-form prox, R(f) = 0.5 ||f||^2
    TOY_PROBLEM      the InverseProblem declaration tying them together
"""

import torch

from core import (
    Feature, features, supports, catalogue,
    ForwardOperator, Objective,
    InverseProblem, DataMode,
)

## float64 so conjugate gradient and the closed forms agree to tight tolerances:
DTYPE = torch.float64
torch.manual_seed(0)


# ── the toy physics ───────────────────────────────────────────────────────────

class _ToyOperator(ForwardOperator):
    """H is an explicit (m, n) matrix, so apply/adjoint are matrix products."""

    name = "toy"
    features = features(Feature.TWO_D)

    def __init__(self, matrix):
        self.H = matrix

    @classmethod
    def from_config(cls, config):
        # what `InverseProblem.make_operator` calls when no build_operator is given:
        m, n = config["toy"]["m"], config["toy"]["n"]
        generator = torch.Generator().manual_seed(config["toy"]["seed"])
        return cls(torch.randn(m, n, generator=generator, dtype=DTYPE))

    def apply(self, f):
        return self.H @ f

    def adjoint(self, y):
        return self.H.T @ y


class _Gaussian:
    """D(Hf, g) = 0.5 ||Hf - g||^2 — the v1 DataFidelity interface."""
    display_name = "gaussian"

    def loss(self, Hf, g):
        return 0.5 * ((Hf - g) ** 2).sum()

    def quadratic_scale(self, n_pixels):
        return 1.0


class _L2:
    """R(f) = 0.5 ||f||^2, whose prox of w*R is f / (1 + w) — the v1 Regularization interface."""
    display_name = "l2"

    def loss(self, f, diff_ops):
        return 0.5 * (f ** 2).sum()

    def prox(self, f, weight, diff_ops):
        return f / (1.0 + weight)


class _NoProx:
    """A regularization WITHOUT a prox, to check the failure is loud and actionable."""
    display_name = "no-prox"

    def loss(self, f, diff_ops):
        return f.abs().sum()


# ── the declaration: a complete inverse problem, in one object ────────────────

TOY_PROBLEM = InverseProblem(
    name="toy",
    operator_class=_ToyOperator,
    load_measurement=lambda config: torch.tensor(config["data"]["g"], dtype=DTYPE),
    load_truth=lambda config: torch.tensor(config["data"]["f_true"], dtype=DTYPE),
    validate=lambda config: ([] if config.get("toy") else ["toy: section missing"]),
    save_image=lambda image, path: None,
    load_image=lambda path: None,
    image_extension="pt",
    description="A tiny dense-matrix problem used to test the core contracts.",
)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_features():
    """Features are typed, set-like, and reject typos immediately."""
    assert Feature.THREE_D == "3d", "str-backed: v1 string comparisons keep working"
    assert str(Feature.ANISOTROPIC) == "anisotropic"

    matirf_like = features(Feature.THREE_D, Feature.ANISOTROPIC)
    assert features("3d", "anisotropic") == matirf_like, "strings and members are interchangeable"

    # the matching rule of the whole framework:
    assert supports(set(), matirf_like), "requiring nothing applies everywhere"
    assert supports({Feature.THREE_D}, matirf_like)
    assert not supports({Feature.TWO_D}, matirf_like)

    # the v1 failure mode — a silent typo in a string set — is now an explicit error:
    try:
        features("3D")            # wrong case
        raise AssertionError("a typo must raise")
    except ValueError as e:
        assert "Unknown feature" in str(e)

    assert "reconstruction space is 3D" in catalogue()
    print("  features        typed, matching rule, typo rejected")


def test_operator():
    """apply/adjoint are enough: every other operation has a correct default."""
    H = torch.randn(6, 4, dtype=DTYPE)
    op = _ToyOperator(H)
    f = torch.randn(4, dtype=DTYPE)
    y = torch.randn(6, dtype=DTYPE)

    assert torch.allclose(op.apply(f), H @ f)
    assert torch.allclose(op.gram(f), H.T @ (H @ f))

    # the adjoint self-check: the single most valuable test for a new operator
    assert op.check_adjoint(f, y) < 1e-10

    # ridge_inverse (conjugate gradient, matrix-free) vs the closed form
    lam = 0.7
    expected = torch.linalg.solve(H.T @ H + lam * torch.eye(4, dtype=DTYPE), H.T @ y)
    assert torch.allclose(op.ridge_inverse(y, lam, n_iter=200, tol=1e-14), expected, atol=1e-8), \
        "conjugate gradient must reproduce (H^T H + lam I)^-1 H^T y"

    # lipschitz (power iteration) vs the largest singular value squared
    expected_L = float(torch.linalg.svdvals(H)[0] ** 2)
    assert abs(op.lipschitz(f, n_iter=500) - expected_L) / expected_L < 1e-6

    # a broken adjoint is caught rather than silently corrupting every gradient solver:
    class _Broken(_ToyOperator):
        def adjoint(self, y):
            return (self.H.T @ y) * 2.0
    try:
        _Broken(H).check_adjoint(f, y)
        raise AssertionError("a wrong adjoint must be detected")
    except AssertionError as e:
        assert "not the transpose" in str(e)

    print("  operator        adjoint check, CG ridge inverse, lipschitz, broken adjoint caught")


def test_objective():
    """L(f) = D(Hf, g) + lambda * R(f), with autograd gradient and a proximal interface."""
    op = _ToyOperator(torch.randn(6, 4, dtype=DTYPE))
    g = torch.randn(6, dtype=DTYPE)
    f = torch.randn(4, dtype=DTYPE)

    # unregularized: L is exactly the data term, untouched (bit-for-bit what v1 computed)
    plain = Objective(op, g, _Gaussian())
    assert not plain.is_regularized and plain.data_weight == 1.0
    assert torch.allclose(plain.value(f), 0.5 * ((op.apply(f) - g) ** 2).sum())
    assert torch.allclose(plain.residual(f), op.apply(f) - g)
    assert torch.allclose(plain.prox_reg(f, 1.0), f), "prox of the zero function is identity"

    # regularized: the v1 blend convention  (1 - lam) * D + lam * R
    lam = 0.25
    obj = Objective(op, g, _Gaussian(), _L2(), lambda_reg=lam)
    assert obj.data_weight == 1.0 - lam
    assert torch.allclose(obj.value(f), (1 - lam) * obj.data_term(f) + lam * obj.reg_term(f))

    # gradient by autograd, against the analytic one: (1-lam) H^T(Hf - g) + lam * f
    analytic = (1 - lam) * op.adjoint(op.apply(f) - g) + lam * f
    assert torch.allclose(obj.grad(f), analytic, atol=1e-10)
    assert not f.requires_grad, "grad() must not disturb the tensor it is given"

    # proximal interface
    assert torch.allclose(obj.prox_reg(f, 2.0), f / (1.0 + 2.0 * lam))

    # a prox-less regularizer must fail at setup, loudly, not produce a wrong result
    try:
        Objective(op, g, _Gaussian(), _NoProx(), lambda_reg=0.1).prox_reg(f)
        raise AssertionError("a missing prox must raise")
    except NotImplementedError as e:
        assert "proximal solver" in str(e)

    # lambda_reg is a blend: anything outside [0, 1] is refused at construction
    for bad in (-1.0, 1.5):
        try:
            Objective(op, g, _Gaussian(), _L2(), lambda_reg=bad)
            raise AssertionError(f"lambda_reg={bad} must raise")
        except ValueError as e:
            assert "[0, 1]" in str(e)

    # the two extremes of the blend behave as the convention says
    pure_data = Objective(op, g, _Gaussian(), _L2(), lambda_reg=0.0)
    assert torch.allclose(pure_data.value(f), plain.value(f)), "lam=0 is pure data fidelity"
    pure_prior = Objective(op, g, _Gaussian(), _L2(), lambda_reg=1.0)
    assert torch.allclose(pure_prior.value(f), pure_prior.reg_term(f)), "lam=1 ignores the data"

    assert "gaussian" in obj.describe() and "l2" in obj.describe()
    print("  objective       v1 blend convention, autograd grad, prox, missing prox")


def test_problem():
    """One declaration answers: what kind of problem, is the config valid, what do I solve."""
    config = {
        "toy": {"m": 6, "n": 4, "seed": 3},
        "input-paths": {"mode": DataMode.REAL.value},
        "data": {"g": [0.0] * 6, "f_true": [1.0] * 4},
    }

    # 1. what kind of problem — read from the operator, declared nowhere else
    assert TOY_PROBLEM.features == features(Feature.TWO_D)
    assert TOY_PROBLEM.supports_synthetic

    # 2. is the config usable
    assert TOY_PROBLEM.validate(config) == []
    assert TOY_PROBLEM.validate({}) == ["toy: section missing"]

    # 3. what am I solving — REAL: g comes from disk, there is no truth
    real = TOY_PROBLEM.prepare(config)
    assert real.mode is DataMode.REAL and not real.has_truth and real.f_true is None
    assert real.g.shape == (6,)
    assert isinstance(real.operator, _ToyOperator), "built via operator_class.from_config"

    # SYNTHETIC: the truth is loaded and the measurement is simulated as g = H f_true
    config["input-paths"]["mode"] = DataMode.SYNTHETIC.value
    synth = TOY_PROBLEM.prepare(config)
    assert synth.mode is DataMode.SYNTHETIC and synth.has_truth
    assert torch.allclose(synth.g, synth.operator.apply(synth.f_true))

    # an absent mode key defaults to REAL rather than crashing (reset-robustness)
    assert DataMode.from_config({}) is DataMode.REAL

    # a problem with no truth loader refuses synthetic mode with an explicit message
    real_only = InverseProblem(
        name="real-only", operator_class=_ToyOperator,
        load_measurement=TOY_PROBLEM.load_measurement, validate=lambda c: [],
        save_image=lambda i, p: None, load_image=lambda p: None, image_extension="pt",
    )
    assert not real_only.supports_synthetic
    try:
        real_only.prepare(config)
        raise AssertionError("synthetic mode must be refused")
    except ValueError as e:
        assert "does not support synthetic mode" in str(e)

    print("  problem         features from operator, validate, prepare in both modes")


def test_end_to_end():
    """The four objects compose: declare a problem, build an objective, minimize it."""
    config = {
        "toy": {"m": 12, "n": 5, "seed": 7},
        "input-paths": {"mode": DataMode.SYNTHETIC.value},
        "data": {"f_true": [1.0, 2.0, 3.0, 4.0, 5.0], "g": []},
    }
    prepared = TOY_PROBLEM.prepare(config)
    objective = Objective(prepared.operator, prepared.g, _Gaussian(), _L2(), lambda_reg=1e-6)

    # a plain gradient descent — note it never mentions the problem, only the objective.
    # This is the whole point: this loop is a solver, and it works for ANY problem.
    step = 1.0 / objective.operator.lipschitz(prepared.f_true)
    f = torch.zeros_like(prepared.f_true)
    for _ in range(4000):
        f = f - step * objective.grad(f)

    error = (f - prepared.f_true).norm() / prepared.f_true.norm()
    assert error < 1e-3, f"gradient descent should recover f_true, relative error {error:.2e}"
    print(f"  end-to-end      solver recovered f_true (relative error {error:.1e})")


def test_noise():
    """
    The Poisson-Gaussian noise model: its STATISTICS, not just that it runs.

    Checked on large constant images, where the theory gives exact targets:
        read noise      variance = sigma^2, whatever the signal
        photon noise    variance = signal * scale / N   (it GROWS with the signal)
        both            the two variances add
    """
    from core import noise

    n = 200_000
    flat = lambda level: torch.full((n,), level, dtype=DTYPE)

    # everything off -> the measurement is returned untouched (no `if` needed by callers)
    g = torch.rand(100, dtype=DTYPE)
    assert noise.add_noise_to_measurement(g, {}) is g
    assert not noise.noise_model({"poisson_noise": False, "gaussian_noise": False}).enabled

    # 100% Gaussian: variance sigma^2, independent of the level
    sigma = 0.05
    for level in (0.2, 0.9):
        out = noise.add_noise_to_measurement(flat(level), {"gaussian_noise": True, "sigma": sigma})
        assert abs(float(out.mean()) - level) < 1e-3
        assert abs(float(out.var()) / sigma ** 2 - 1) < 0.02

    # 100% Poisson: variance proportional to the signal (scale = max = 1 here)
    photons = 50.0
    signal = torch.cat([flat(0.25), flat(1.0)])
    out = noise.add_noise_to_measurement(signal, {"poisson_noise": True, "photons": photons})
    dim, bright = out[:n], out[n:]
    assert abs(float(bright.mean()) - 1.0) < 1e-2, "photon noise must be unbiased"
    assert abs(float(dim.var()) / (0.25 / photons) - 1) < 0.03
    assert abs(float(bright.var()) / (1.0 / photons) - 1) < 0.03
    ## the defining property: brighter is noisier in absolute terms, cleaner in relative terms
    assert float(bright.var()) > float(dim.var())
    assert float(bright.std() / bright.mean()) < float(dim.std() / dim.mean())

    # Poisson-Gaussian: the two variances add
    both = noise.add_noise_to_measurement(
        signal, {"poisson_noise": True, "photons": photons, "gaussian_noise": True, "sigma": sigma})
    assert abs(float(both[n:].var()) / (1.0 / photons + sigma ** 2) - 1) < 0.03

    # the v1 failure: Poisson on a [0, 1] image without a photon count was binary speckle.
    # With a photon count, the noisy image stays a faithful, graded version of the signal.
    assert float(out.unique().numel()) > 20, "photon noise must not collapse to 0/1 values"

    # reproducibility: same seed, same draw; another seed, another draw
    config = {"gaussian_noise": True, "sigma": 0.1, "seed": 3}
    a = noise.add_noise_to_measurement(g, config)
    b = noise.add_noise_to_measurement(g, config)
    c = noise.add_noise_to_measurement(g, {**config, "seed": 4})
    assert torch.equal(a, b) and not torch.equal(a, c)

    # a v1 config keeps working, and gives the Gaussian noise v1 actually produced
    legacy = noise.noise_model({"add_noise": True, "is_gaussian": False, "sigma": 0.02})
    assert legacy.sigma == 0.02 and legacy.photons is None
    assert not noise.noise_model({"add_noise": False, "sigma": 0.02}).enabled

    # validation: one readable message per unusable setting
    assert noise.validate({}) == []
    assert noise.validate({"gaussian_noise": True, "sigma": "None"}) == [
        "Read noise: sigma is required when it is enabled"]
    assert noise.validate({"poisson_noise": True, "photons": 0}) == [
        "Photon noise: the photon count must be positive (got 0)"]
    assert noise.validate({"poisson_noise": True}) == [
        "Photon noise: the photon count is required when it is enabled"]

    # the (a, b) every other part of the framework uses: a = 1/N, b = sigma^2
    both_model = noise.noise_model({"poisson_noise": True, "photons": 50.0,
                                    "gaussian_noise": True, "sigma": 0.05})
    assert both_model.a == 1 / 50.0 and abs(both_model.b - 0.0025) < 1e-15

    # g >= 0 always: read noise on a dark pixel is clipped (option A), photon noise never
    # needs it
    dark = noise.add_noise_to_measurement(flat(0.0), {"gaussian_noise": True, "sigma": 0.05})
    assert float(dark.min()) >= 0.0, "the noisy measurement must stay non-negative"
    assert 0.3 * 0.05 < float(dark.mean()) < 0.5 * 0.05, "the documented ~0.4 sigma bias"

    # the formula line follows the ticked boxes
    assert r"\mathcal{P}" in noise.formula({"poisson_noise": True, "photons": 100})
    assert r"\mathcal{P}" not in noise.formula({"gaussian_noise": True, "sigma": 0.01})
    assert "b = 0" in noise.formula({"poisson_noise": True, "photons": 100})
    assert "a = 0" in noise.formula({"gaussian_noise": True, "sigma": 0.01})
    assert "no noise" in noise.formula({})
    print("  noise           Gaussian var = sigma^2, Poisson var = signal/N, they add, seeded")


def test_noise_estimation():
    """
    `estimate` recovers (a, b) from the noisy image alone.

    On a smooth image — which is what a blurred measurement g = H f is — within 10 %; the
    part the model does not have comes back as exactly 0.
    """
    from core import noise
    y, x = torch.meshgrid(torch.linspace(0, 1, 384, dtype=DTYPE),
                          torch.linspace(0, 1, 384, dtype=DTYPE), indexing="ij")
    smooth = 0.1 + 0.9 * torch.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.08)

    cases = [
        ({"gaussian_noise": True, "sigma": 0.02}, False, True),
        ({"poisson_noise": True, "photons": 200}, True, False),
        ({"poisson_noise": True, "photons": 500, "gaussian_noise": True, "sigma": 0.01}, True, True),
    ]
    for config, poisson, gaussian in cases:
        truth = noise.noise_model(config)
        a, b = noise.estimate(noise.add_noise_to_measurement(smooth, config), poisson, gaussian)
        for name, true, got in (("a", truth.a, a), ("b", truth.b, b)):
            if true == 0:
                assert got == 0, f"{name} is not in the model and must come back as 0"
            else:
                assert abs(got / true - 1) < 0.10, f"{config}: {name} = {got:.3g}, true {true:.3g}"

    # a noiseless image gives (0, 0), not a division by zero
    assert noise.estimate(smooth) == (0.0, 0.0) or max(noise.estimate(smooth)) < 1e-9
    assert noise.estimate(torch.full((32, 32), 0.5, dtype=DTYPE)) == (0.0, 0.0)
    print("  estimation      (a, b) recovered within 10 % on a smooth image; absent part = 0")


def test_normalization():
    """Every method gives g >= 0; all but min-max are a pure scale, which keeps g = H f linear."""
    from core import normalization as norm

    g = torch.rand(4, 30, 30, dtype=DTYPE) * 7.0
    g[0, 0, 0] = -0.3                          # what background subtraction may leave
    for method in norm.NORMALIZATIONS:
        out = norm.normalize(g, method)
        assert float(out.min()) >= 0, f"{method}: negative output"
        assert method in norm.formula(method) or norm.formula(method).startswith("g")

    assert float(norm.normalize(g, "peak").max()) == 1.0
    assert float(norm.normalize(g, "min-max").min()) == 0.0

    # a pure scale: normalizing H f and normalizing 3 * H f give the same image
    for method in ("peak", "percentile 99.9", "L2 energy", "RMS", "mean"):
        assert torch.allclose(norm.normalize(g.clamp(min=0), method),
                              norm.normalize(3.0 * g.clamp(min=0), method)), method
    # min-max is not: it subtracts an offset (why it is not the default)
    shifted = g.clamp(min=0) + 1.0
    assert float(norm.normalize(shifted, "min-max").min()) == 0.0
    assert float(norm.normalize(shifted, "peak").min()) > 0.1

    # percentile: a hot pixel saturates instead of squashing the rest of the image
    hot = torch.rand(100, 100, dtype=DTYPE)
    hot[0, 0] = 1000.0
    assert float(norm.normalize(hot, "percentile 99.9").median()) > 0.3
    assert float(norm.normalize(hot, "peak").median()) < 0.01

    # the default, and a v1 config's integer
    assert norm.resolve(None) == "peak" and norm.resolve(1) == "min-max"
    assert norm.validate("peak") == [] and norm.validate("bogus")
    print("  normalization   >= 0 always, a pure scale except min-max, hot pixels saturate")


def main():
    print("core — contract tests\n")
    test_features()
    test_operator()
    test_objective()
    test_problem()
    test_end_to_end()
    test_noise()
    test_noise_estimation()
    test_normalization()
    print("\nAll core contracts hold.")


if __name__ == "__main__":
    main()
