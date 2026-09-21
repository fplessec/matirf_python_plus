"""
Tests of the Hessian regularizations — Hessian-Frobenius and SHV.

Run it:
    python -m solvers.regularizers._tests

Three kinds of check:

    REFERENCE    SHV must reproduce SPITFIRe (S. Prigent et al.). Its `hv_loss` and
                 `hv_loss_3d` are copied below VERBATIM and compared on random images.
    DEFINITION   Hessian-Frobenius must be the standard sum_x ||H f(x)||_F: zero on linear
                 images, the exact analytic value on quadratic ones — and, unlike v1, not
                 fooled by a curved image whose Hessian entries cancel.
    PROX         the proximal operator must actually minimise 1/2||u - f||^2 + lam R(u):
                 the adjoint is exact, the output beats the input and every perturbation.
"""

import torch

from solvers.differential_operators import DifferentialOperators
from solvers.regularizers import HessianFrobeniusRegularization, SHVRegularization
from solvers.regularizers import _hessian

DTYPE = torch.float64
torch.manual_seed(0)


# ── SPITFIRe, copied verbatim (sdeconv, S. Prigent) — do not edit ─────────────

def hv_loss(img, weighting):
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    h_v = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                     (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1]))
    return torch.mean(h_v)


def hv_loss_3d(img, delta, weighting):
    img_ = img[:, :, 1:-1, 1:-1, 1:-1]
    d11 = -img[:, :, 1:-1, 1:-1, 2:] + 2*img_ - img[:, :, 1:-1, 1:-1, :-2]
    d22 = -img[:, :, 1:-1, 2:, 1:-1] + 2*img_ - img[:, :, 1:-1, :-2, 1:-1]
    d33 = delta*delta*(-img[:, :, 2:, 1:-1, 1:-1] + 2*img_ - img[:, :, :-2, 1:-1, 1:-1])
    d12_d21 = img[:, :, 1:-1, 2:, 2:] - img[:, :, 1:-1, 1:-1, 2:] - img[:, :, 1:-1, 2:, 1:-1] + img_
    d13_d31 = delta*(img[:, :, 2:, 1:-1, 2:] - img[:, :, 1:-1, 1:-1, 2:]
                     - img[:, :, 2:, 1:-1, 1:-1] + img_)
    d23_d32 = delta*(img[:, :, 2:, 2:, 1:-1] - img[:, :, 1:-1, 2:, 1:-1]
                     - img[:, :, 2:, 1:-1, 1:-1] + img_)

    h_v = torch.square(weighting*d11) + torch.square(weighting*d22) + torch.square(
        weighting*d33) + 2 * torch.square(weighting*d12_d21) + 2 * torch.square(
        weighting*d13_d31) + 2 * torch.square(weighting*d23_d32) + torch.square((1-weighting)*img_)

    return torch.mean(torch.sqrt(h_v))


# ── helpers ───────────────────────────────────────────────────────────────────

def _sum_norm(f, delta, weight):
    """R(f) as a SUM over pixels and without the loss epsilon — what the prox minimises."""
    terms = _hessian.components(f, delta, weight)
    return torch.sqrt(sum(t * t for t in terms)).sum()


def _grid(shape):
    axes = torch.meshgrid(*[torch.arange(n, dtype=DTYPE) for n in shape], indexing="ij")
    return axes


# ── tests ─────────────────────────────────────────────────────────────────────

def test_matches_spitfire():
    """SHV is SPITFIRe's term, stencil for stencil, in 2D and 3D, for several weights."""
    worst = 0.0
    for weight in (0.0, 0.3, 0.6, 1.0):
        img2 = torch.rand(40, 50, dtype=DTYPE)
        ours = SHVRegularization(rho=weight).loss(img2, DifferentialOperators(delta=1.0))
        theirs = hv_loss(img2[None, None], weight)
        worst = max(worst, abs(float(ours - theirs)))

        for delta in (1.0, 0.05, 0.4):
            img3 = torch.rand(9, 20, 24, dtype=DTYPE)
            ours = SHVRegularization(rho=weight).loss(img3, DifferentialOperators(delta=delta))
            theirs = hv_loss_3d(img3[None, None], delta, weight)
            worst = max(worst, abs(float(ours - theirs)))
    ## the only difference is the 1e-12 inside our square root (NaN-safe gradients)
    assert worst < 1e-5, f"SHV differs from SPITFIRe by {worst:.2e}"
    print(f"  SPITFIRe        2D + 3D, 4 weights, 3 deltas: max gap {worst:.1e}")


def test_frobenius_is_the_standard_definition():
    """sum_x ||H f(x)||_F — checked against closed forms, and against v1's failure case."""
    ops = DifferentialOperators(delta=1.0)
    frob = HessianFrobeniusRegularization()

    # SHV with weight 1 IS Hessian-Frobenius
    f = torch.rand(30, 30, dtype=DTYPE)
    assert torch.allclose(frob.loss(f, ops), SHVRegularization(rho=1.0).loss(f, ops))

    # a linear image has a zero Hessian: it costs nothing (the point of second order)
    y, x = _grid((30, 40))
    linear = 3.0 * x - 2.0 * y + 5.0
    assert float(frob.loss(linear, ops)) < 1e-5, "a linear image must be free"

    # a quadratic image has a constant Hessian; the stencils are exact on quadratics:
    #   f = a x^2 + b y^2 + c x y  ->  fxx = 2a, fyy = 2b, fxy = c
    #   ||H||_F = sqrt(4a^2 + 4b^2 + 2c^2)
    a, b, c = 0.7, -0.3, 0.5
    quadratic = a * x * x + b * y * y + c * x * y
    expected = (4 * a * a + 4 * b * b + 2 * c * c) ** 0.5
    assert abs(float(frob.loss(quadratic, ops)) - expected) < 1e-6

    # v1's definition summed the SIGNED entries: on a saddle x^2 - y^2 they cancel to 0,
    # although the image is curved everywhere. The standard norm sees it.
    saddle = x * x - y * y
    v1_value = float(ops.hessian(saddle)[:, :, 2:-2, 2:-2].sum(dim=(0, 1)).mean())
    ours = float(frob.loss(saddle, ops))
    assert abs(v1_value) < 1e-9 and abs(ours - 8 ** 0.5) < 1e-6
    print(f"  definition      linear -> 0, quadratic exact, saddle x^2-y^2: "
          f"v1 {v1_value:.1f} vs {ours:.3f}")


def test_adjoint_is_exact():
    """
    The hand-written L^T must be the true adjoint — the prox is only correct if it is.

    Two independent certificates: the inner-product identity <L u, p> == <u, L^T p>, and
    equality with autograd, which computes L^T p = d/du <L u, p> exactly for a linear L.
    """
    for shape, delta in (((25, 31), 1.0), ((7, 15, 18), 0.05)):
        for weight in (1.0, 0.6):
            u = torch.randn(*shape, dtype=DTYPE)
            p = [torch.randn_like(t) for t in _hessian.components(u, delta, weight)]
            hand = _hessian.adjoint(p, u, delta, weight)

            left = sum((t * q).sum() for t, q in zip(_hessian.components(u, delta, weight), p))
            right = (u * hand).sum()
            assert abs(float(left - right)) / abs(float(left)) < 1e-12

            z = torch.zeros_like(u, requires_grad=True)
            inner = sum((t * q).sum() for t, q in zip(_hessian.components(z, delta, weight), p))
            (automatic,) = torch.autograd.grad(inner, z)
            assert torch.allclose(hand, automatic, atol=1e-12), "hand adjoint != autograd"
    print("  adjoint         hand-written L^T == autograd, and <Lu,p> == <u,L^T p>, 2D and 3D")


def test_operator_norm_bound():
    """The step 1/||L||^2 must come from a TRUE upper bound, or the iteration may diverge."""
    for shape, delta, weight in (((40, 40), 1.0, 1.0), ((9, 30, 30), 0.3, 0.6)):
        x = torch.randn(*shape, dtype=DTYPE)
        for _ in range(200):                                   # power iteration on L^T L
            x = _hessian.adjoint(_hessian.components(x, delta, weight), x, delta, weight)
            x = x / x.norm()
        measured = float((_hessian.adjoint(_hessian.components(x, delta, weight),
                                           x, delta, weight) * x).sum())
        bound = _hessian.operator_norm_squared(len(shape), delta, weight)
        assert measured <= bound, f"bound {bound} below the true norm {measured}"
    print(f"  norm bound      analytic bound >= measured ||L||^2")


def test_prox_minimises():
    """The prox output must beat the input and every small perturbation of itself."""
    for shape, delta, weight in (((32, 32), 1.0, 1.0), ((32, 32), 1.0, 0.6),
                                 ((8, 20, 20), 0.2, 1.0), ((8, 20, 20), 0.2, 0.6)):
        f = torch.rand(*shape, dtype=DTYPE)
        lam = 0.05

        def objective(u):
            return 0.5 * float(((u - f) ** 2).sum()) + lam * float(_sum_norm(u, delta, weight))

        u = _hessian.prox(f, lam, delta, weight, n_iter=300)
        best = objective(u)
        assert best < objective(f), "the prox must improve on its input"
        for _ in range(20):
            nudge = 1e-3 * torch.randn_like(u)
            assert objective(u + nudge) >= best - 1e-9, "a perturbation beat the prox"
    print("  prox            beats its input and 20 random perturbations, 2D/3D, w=1 and 0.6")


def test_prox_leaves_linear_images_alone():
    """R(linear) = 0 for Hessian-Frobenius, so its prox must return a linear image unchanged."""
    y, x = _grid((30, 30))
    linear = 0.3 * x + 0.1 * y
    out = HessianFrobeniusRegularization(n_iter=50).prox(linear, 0.5, DifferentialOperators())
    assert torch.allclose(out, linear, atol=1e-8)
    print("  prox            linear image returned unchanged")


def test_gradients_are_finite():
    """Adam differentiates the loss; zero regions (positivity clamp) must not give NaN."""
    f = torch.rand(8, 20, 20, dtype=torch.float32)
    f[:, :10] = 0.0                       # what a positivity-projected reconstruction looks like
    for reg in (HessianFrobeniusRegularization(), SHVRegularization()):
        x = f.clone().requires_grad_(True)
        reg.loss(x, DifferentialOperators(delta=0.05)).backward()
        assert torch.isfinite(x.grad).all(), f"{reg.name}: non-finite gradient"
    print("  autograd        finite gradients, including over zero regions")


def main():
    print("solvers.regularizers — Hessian-Frobenius and SHV\n")
    test_matches_spitfire()
    test_frobenius_is_the_standard_definition()
    test_adjoint_is_exact()
    test_operator_norm_bound()
    test_prox_minimises()
    test_prox_leaves_linear_images_alone()
    test_gradients_are_finite()
    print("\nThe Hessian regularizations match their definitions.")


if __name__ == "__main__":
    main()
