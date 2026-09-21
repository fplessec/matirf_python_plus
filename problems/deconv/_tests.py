"""
Test and reference implementation of the deconvolution problem.

Run it:
    python -m problems.deconv._tests

Same two kinds of check as MA-TIRF: a REGRESSION against the v1 implementation that still
ships alongside, and the CONTRACTS every solver relies on. The interesting one here is the
Wiener closed form — an exact inverse that the base class could only approximate.
"""

import json
from pathlib import Path

import torch

from fileio import load_json, load_png
from core import DataMode, Feature, features, Objective
from core.operator import ForwardOperator
from problems.deconv import DECONV_MEASUREMENTS_DIR
from solvers import SOLVERS, Adam

from problems.deconv import DECONV, DeconvOperator
from problems.deconv.operator import gaussian_psf

PNG = str(DECONV_MEASUREMENTS_DIR / "img_001.png")
JSON = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")


def _config(mode=DataMode.SYNTHETIC, add_noise=None):
    return {
        "input-paths": {"mode": mode.value, "png": PNG, "json": JSON},
        "add-noise": add_noise or {},
        "algo-params": {},
    }


# ── regression against v1 ─────────────────────────────────────────────────────

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


def test_matches_v1():
    """
    The PSF and the convolution must still produce what v1 produced.

    Compared against outputs recorded before the v1 implementation was deleted, so the
    guarantee outlives the code it was written against. The PSF is small enough to store
    whole; the blurred images are stored as a fingerprint (see problems/matirf/_tests.py).
    """
    reference = json.loads((Path(__file__).parent / "_v1_reference.json").read_text())

    params = load_json(JSON)
    psf = gaussian_psf(params["sigma"], params["kernel_size"])
    assert torch.allclose(psf, torch.tensor(reference["psf"], dtype=psf.dtype), atol=1e-8), \
        "the PSF changed"

    operator = DeconvOperator(psf, params)
    image = load_png(PNG)
    _assert_fingerprint(_fingerprint(operator.apply(image)), reference["blurred"], "blur")
    _assert_fingerprint(_fingerprint(operator.adjoint(image)), reference["correlated"], "adjoint")

    # an even kernel size is still forced odd, as v1 did
    assert gaussian_psf(2.0, 8).shape == (9, 9)
    assert abs(float(gaussian_psf(3.0, 15).sum()) - 1.0) < 1e-6, "a PSF must preserve intensity"
    print("  regression      PSF, blur and adjoint unchanged since v1")


# ── the operator contract ─────────────────────────────────────────────────────

def test_operator_contract():
    """What every solver assumes, plus the Wiener closed form."""
    operator = DeconvOperator.from_config(_config())
    assert operator.features == features(Feature.TWO_D)

    f = torch.rand(64, 64, dtype=operator.psf.dtype)
    y = torch.rand(64, 64, dtype=operator.psf.dtype)
    assert operator.apply(f).shape == f.shape, "blurring preserves the shape"
    assert operator.check_adjoint(f, y) < 1e-4, "adjoint must be the true transpose"

    # the Wiener form must agree with the base class's conjugate gradient
    b = torch.rand(64, 64, dtype=operator.psf.dtype)
    lam = 0.05
    closed = operator.solve_normal(b, lam)
    by_cg = ForwardOperator.solve_normal(operator, b, lam, n_iter=400, tol=1e-12)
    relative = ((closed - by_cg).abs().max() / closed.abs().max()).item()
    assert relative < 1e-3, f"Wiener and CG disagree by {relative:.2e} (relative)"

    # and it must actually invert: solve_normal(gram(f) + lam f, lam) == f
    recovered = operator.solve_normal(operator.gram(f) + lam * f, lam)
    assert torch.allclose(recovered, f, atol=1e-4), "the closed form must be a true inverse"

    # lam = 0 on a Gaussian PSF is the naive inverse filter: it must not produce infinities
    assert torch.isfinite(operator.solve_normal(b, 0.0)).all()

    # the spectrum cache is keyed by shape, so a second image size is handled correctly
    small = torch.rand(32, 32, dtype=operator.psf.dtype)
    assert operator.apply(small).shape == (32, 32)
    assert set(operator._spectra) == {(64, 64), (32, 32)}
    print("  operator        true adjoint, Wiener == CG and is an exact inverse, cache by shape")


# ── the declaration ───────────────────────────────────────────────────────────

def test_problem_declaration():
    assert DECONV.features == features(Feature.TWO_D)
    assert DECONV.supports_synthetic and DECONV.image_extension == "png"
    assert DECONV.validate(_config()) == []
    assert DECONV.validate({}) == ["Input file (PNG): not provided",
                                   "PSF parameters file (JSON): not provided"]

    synthetic = DECONV.prepare(_config(DataMode.SYNTHETIC))
    assert synthetic.has_truth and synthetic.g.shape == synthetic.f_true.shape
    assert not torch.allclose(synthetic.g, synthetic.f_true), "the measurement must be blurred"

    real = DECONV.prepare(_config(DataMode.REAL))
    assert not real.has_truth and real.f_true is None

    noisy = DECONV.prepare(_config(DataMode.SYNTHETIC,
                                   {"gaussian_noise": True, "sigma": 0.05}))
    clean = DECONV.prepare(_config(DataMode.SYNTHETIC))
    assert (noisy.g - clean.g).abs().mean() > 1e-3, "the configured noise must be applied"
    print("  declaration     validation, both modes, noise applied when configured")


# ── every solver, on this problem ─────────────────────────────────────────────

def test_all_solvers_run():
    """Every solver reconstructs this problem without a line of deconvolution-specific code."""
    prepared = DECONV.prepare(_config(DataMode.SYNTHETIC))
    ## a crop keeps the test quick; the operator is shape-agnostic
    f_true = prepared.f_true[:96, :96]
    operator = prepared.operator
    g = operator.apply(f_true)

    class _Gaussian:
        display_name = "gaussian"
        def loss(self, Hf, gg):
            return 0.5 * ((Hf - gg) ** 2).sum()
        def quadratic_scale(self, n_pixels):   # 1/2 ||r||^2: a quadratic of weight 1
            return 1.0

    budgets = {
        "ADAM": {"max_iter": 60, "lr": 0.02, "K": 60, "EPS": 1e-14},
        "PPXA": {"max_iter": 40, "lambda_relax": 1.0, "gamma": 1.0, "K": 40, "EPS": 1e-14},
        "ADMM": {"iter": 15, "mu": 0.05, "threshold_ratio": 0.0},
        "PNP": {"iter": 6, "sigma": 5.0, "denoiser": "None", "kai_zhang": True},
        "ADMM-PnP": {"iter": 15, "rho": 0.05, "sigma": 5.0, "denoiser": "None"},
        "MCMC": {"max_iter": 40, "beta": 1e-4, "sigma": 0.005, "K": 40, "lambda_rr": 0.05},
        "MCMCv2": {"max_iter": 40, "sigma": 0.02, "K": 40, "denoiser": "TV Bregman"},
    }

    baseline = (operator.adjoint(g) - f_true).norm() / f_true.norm()
    for name, solver_class in SOLVERS.items():
        torch.manual_seed(5)
        objective = Objective(operator, g, _Gaussian())
        solver = solver_class()
        params = budgets[name]
        f = solver.solve(objective, solver.initial_guess(objective, params), params)
        error = ((f - f_true).norm() / f_true.norm()).item()
        assert torch.isfinite(f).all(), f"{name} produced non-finite values"
        assert error < baseline, (f"{name}: error {error:.3e} is no better than the "
                                 f"back-projection ({baseline:.3e})")
    print(f"  all solvers     {len(SOLVERS)} solvers reconstructed a real image, none deconv-specific")


def test_wiener_and_the_regularization_tradeoff():
    """
    Wiener filtering inverts the blur in one FFT round trip — and shows why lam matters.

    `ridge_inverse` is `solve_normal` applied to H^t g, so on this problem it is a single
    pointwise division in Fourier. Sweeping lam over the real image exhibits the central
    tradeoff of every regularized inverse problem, which is worth seeing once concretely:

        lam too large   the solution is pulled toward zero and stays blurred
        lam just right  the best restoration
        lam too small   the PSF's spectrum is near zero at high frequency, so dividing by
                        it amplifies whatever is there — rounding error if the data is
                        clean, actual noise if it is not — and the result diverges

    And the optimum MOVES with the noise: the noisier the measurement, the more
    regularization is needed. That is why lambda_rr is a parameter and not a constant.
    """
    def error(estimate, truth):
        return ((estimate - truth).norm() / truth.norm()).item()

    # ── clean data ───────────────────────────────────────────────────────────
    clean = DECONV.prepare(_config(DataMode.SYNTHETIC))
    f_true, operator, g = clean.f_true, clean.operator, clean.g

    blurred = error(g, f_true)
    best = error(operator.ridge_inverse(g, 1e-5), f_true)
    over_regularized = error(operator.ridge_inverse(g, 1e-1), f_true)
    under_regularized = error(operator.ridge_inverse(g, 1e-8), f_true)

    assert best < 0.7 * blurred, f"a well-chosen lam should clearly deblur ({best:.4f} vs {blurred:.4f})"
    assert over_regularized > best, "too much regularization leaves the image blurred"
    assert under_regularized > blurred, (
        "too little regularization must visibly diverge — this is the ill-posedness, "
        f"got {under_regularized:.4f}")

    # ── the same image, now noisy ────────────────────────────────────────────
    noisy = DECONV.prepare(_config(DataMode.SYNTHETIC,
                                   {"gaussian_noise": True, "sigma": 0.01}))
    gn = noisy.g
    noisy_small_lam = error(operator.ridge_inverse(gn, 1e-5), f_true)
    noisy_large_lam = error(operator.ridge_inverse(gn, 1e-2), f_true)
    assert noisy_large_lam < noisy_small_lam, (
        "with noise the optimal lam moves UP: the value that was best on clean data "
        f"({noisy_small_lam:.4f}) is now far worse than a larger one ({noisy_large_lam:.4f})")

    print(f"  wiener          clean: {blurred:.4f} blurred -> {best:.4f} at lam=1e-5 "
          f"(diverges to {under_regularized:.2f} at lam=1e-8)")
    print(f"  tradeoff        with noise the best lam moves up: "
          f"{noisy_small_lam:.4f} at 1e-5 vs {noisy_large_lam:.4f} at 1e-2")


def main():
    print("problems.deconv — regression against v1, and contracts\n")
    test_matches_v1()
    test_operator_contract()
    test_problem_declaration()
    test_all_solvers_run()
    test_wiener_and_the_regularization_tradeoff()
    print("\nDeconvolution is ported faithfully.")


if __name__ == "__main__":
    main()
