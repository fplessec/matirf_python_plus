"""
lambda interpretability study -- the evidence behind docs/algorithms/lambda_interpretability.md (D1).

Run it from the repository root:

    env/bin/python -m benchmarks.studies.lambda_interpretability

The shared objective is L = (1 - l) D + l R with l in [0, 1]. D is a scaled per-pixel
negative log-likelihood (about 1/2 at the truth, so D = O(1) whatever the problem); R is a
mean of gradients / curvatures with no intrinsic scale. So R is 2-3 orders of magnitude
below D, and l is a *dead knob* until it is within a hair of 1 -- not interpretable, not
transferable.

Calibrating "R onto D" fixes this: replace R by R / R_ref with a single scalar R_ref of the
right magnitude, so the two terms are comparable and l reads as a regularization share. The
share ruler is r(l) = 1 - R(f_l) / R(f0), f0 = the l=0 reconstruction (the metric
`regularization_share`, benchmarks/metrics.py); it is never calibrated (that would be
circular). The reference R_ref is the only free choice; the menu:

    C0  none         kappa = 1                (standard L, the dead knob, control)
    C1  R(f0)        the l=0 reconstruction   (principled = the r denominator; needs a pre-pass)
    C2  R(f_init)    the ridge start          (already computed -> free, deterministic)
    C4  R(f_true)    the truth                (oracle, benchmark-only, an upper bound)

Three parts, on MA-TIRF (ill-posed) and deconvolution (well-posed), real oracle noise:
    PART 1  l is a dead knob under standard L; calibrating (C2) grades it.
    PART 2  the reference choice: C2 (free) tracks the oracle C4 and keeps a stable anchor.
    PART 3  the ridge-start weight: s2^2 (the default) vs a lighter / heavier weight.

Decision recorded in the note: B with C2. All solves start from the s2^2 ridge start (the
solver default), under oracle noise (D uses the exact sigma^2 that was injected).
"""
import time

import torch

from core import DataMode, Objective
from core.metrics import optimal_scale
from core.normalization import normalize_and_add_noise
from pipeline import resolve_noise_model, noise_parameters
from problems.matirf import MATIRF, MATIRF_SYNTHETIC_DIR
from problems.deconv import DECONV, DECONV_MEASUREMENTS_DIR
from solvers import Adam
from solvers.base import ridge_start, second_eigenvalue, ridge_weight
from solvers.differential_operators import DifferentialOperators
from solvers.regularizers import REGULARIZATION_REGISTRY

ITERS = 200
START = {"init": "ridge"}          # lambda_rr empty = s2^2, the solver default
MATIRF_CROP, DECONV_CROP = 40, 128
JSON_MATIRF = str(MATIRF_SYNTHETIC_DIR / "measurement_parameters.json")
JSON_DECONV = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")


class _Calibrated(Objective):
    """L = (1 - l) D + l * (R / R_ref): R rescaled so the two terms are comparable."""

    def __init__(self, *args, reference, **kwargs):
        super().__init__(*args, **kwargs)
        self._inv = 1.0 / reference

    def reg_term(self, f):
        return super().reg_term(f) * self._inv

    def prox_reg(self, f, tau=1.0):
        return super().prox_reg(f, tau * self._inv)


def _central_crop(t, c):
    h, w = t.shape[-2], t.shape[-1]
    return t[..., (h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c].contiguous()


def _matirf_case(truth, sigma):
    """Operator, cropped g and truth, oracle fidelity. The lateral crop is clean: H acts
    per depth column, so cropping columns leaves the same H on fewer of them."""
    cfg = {"input-paths": {"mode": DataMode.SYNTHETIC.value,
                           "tif": str(MATIRF_SYNTHETIC_DIR / f"{truth}.TIF"), "json": JSON_MATIRF},
           "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": "peak"},
           "add-noise": {"gaussian_noise": True, "sigma": sigma, "seed": 1},
           "noise-model": {"parameters": "oracle"}, "algo-params": {}}
    prep = MATIRF.prepare(cfg)
    op = prep.operator
    fidelity, _ = resolve_noise_model(prep, cfg)
    c = MATIRF_CROP
    return (op, prep.g[..., :c, :c].contiguous(), prep.f_true[..., :c, :c].contiguous(),
            fidelity, float(op.estimate_anisotropy_ratio()))


def _deconv_case(truth, sigma):
    """Crop the truth, then RE-SIMULATE g on the patch (convolution + configured noise) with
    its own oracle fidelity -- convolution couples neighbours, so cropping g would mismatch."""
    cfg = {"input-paths": {"mode": DataMode.SYNTHETIC.value,
                           "png": str(DECONV_MEASUREMENTS_DIR / f"{truth}.png"),
                           "json": JSON_DECONV, "normalization": "peak"},
           "add-noise": {"gaussian_noise": True, "sigma": sigma, "seed": 1},
           "noise-model": {"parameters": "oracle"}, "algo-params": {}}
    prep = DECONV.prepare(cfg)
    op = prep.operator
    f_true = _central_crop(prep.f_true, DECONV_CROP)
    g = normalize_and_add_noise(op.apply(f_true), cfg)
    cls, a, b, _ = noise_parameters(cfg, g)
    return op, g, f_true, cls.from_noise(a, b), 1.0


def _case(problem, truth, sigma):
    return (_matirf_case if problem == "matirf" else _deconv_case)(truth, sigma)


def _tools(op, g, f_true, fidelity, diff):
    def solve(reg, lam, reference, start=None):
        kind = dict(operator=op, g=g, data_fidelity=fidelity, regularization=reg,
                    lambda_reg=lam, diff_ops=diff)
        obj = _Calibrated(**kind, reference=reference) if reference is not None else Objective(**kind)
        start = Adam().initial_guess(obj, START) if start is None else start
        return Adam().solve(obj, start, {"max_iter": ITERS})

    def R_of(reg, f):
        return float(reg.loss(f, diff))

    def nmse(f):
        alpha = optimal_scale(f, f_true)
        return float(((alpha * f - f_true).norm() / f_true.norm()) ** 2)

    return solve, R_of, nmse


LAMBDAS_FINE = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
LAMBDAS_BEST = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]


def part1_interpretability():
    """standard l is a dead knob; calibrating by R(f_init) (C2) grades it."""
    print(f"\n{'#'*84}\n# PART 1 -- l is a dead knob under standard L; C2 calibration grades it")
    print(f"# (sigma = 0.05, ridge s2^2 start; ruler r = 1 - R(f_l)/R(f0))\n{'#'*84}")
    for problem, truth in (("matirf", "fibres"), ("deconv", "img_001")):
        op, g, f_true, fidelity, delta = _case(problem, truth, 0.05)
        diff = DifferentialOperators(delta=delta)
        solve, R_of, nmse = _tools(op, g, f_true, fidelity, diff)
        for reg_name in ("tv", "tikhonov"):
            reg = REGULARIZATION_REGISTRY[reg_name]()
            f0 = solve(reg, 0.0, None)
            R0 = R_of(reg, f0)
            f_init = ridge_start(Objective(operator=op, g=g, data_fidelity=fidelity,
                                           regularization=reg, lambda_reg=0.0, diff_ops=diff), START)
            Rinit = R_of(reg, f_init)
            print(f"\n=== {problem} | {truth} | {reg_name} | R(f0)={R0:.4g}  R(f_init)={Rinit:.4g} ===")
            print(f"{'lambda':>7} | {'r std':>7} {'nmse std':>9} | {'r C2':>7} {'nmse C2':>9}")
            for lam in LAMBDAS_FINE:
                fs = solve(reg, lam, None)
                fc = f0 if lam == 0.0 else solve(reg, lam, Rinit)
                rs = 1 - R_of(reg, fs) / R0
                rc = 1 - R_of(reg, fc) / R0
                print(f"{lam:7.2f} | {rs:7.3f} {nmse(fs):9.3f} | {rc:7.3f} {nmse(fc):9.3f}", flush=True)


def part2_reference_choice():
    """C1 / C2 / C4: best lambda and the C2 anchor across truths, noise and problems."""
    print(f"\n{'#'*84}\n# PART 2 -- the reference choice: best lambda* (nmse*) per reference, and C2's anchor")
    print(f"# C2 (free) vs C1 (pre-pass) vs C4 (oracle, benchmark-only){'#'*10}\n{'#'*84}")
    matrix = [("matirf", t, s) for t in ("vesicles", "fibres", "cell") for s in (0.05, 0.15)]
    matrix += [("deconv", t, s) for t in ("img_001", "img_002", "img_003") for s in (0.05, 0.15)]
    rows = {}
    for problem, truth, sigma in matrix:
        op, g, f_true, fidelity, delta = _case(problem, truth, sigma)
        diff = DifferentialOperators(delta=delta)
        solve, R_of, nmse = _tools(op, g, f_true, fidelity, diff)
        for reg_name in ("tv", "tikhonov"):
            reg = REGULARIZATION_REGISTRY[reg_name]()
            f0 = solve(reg, 0.0, None)
            R0 = R_of(reg, f0)
            f_init = ridge_start(Objective(operator=op, g=g, data_fidelity=fidelity,
                                           regularization=reg, lambda_reg=0.0, diff_ops=diff), START)
            refs = {"C1": R0, "C2": R_of(reg, f_init), "C4": R_of(reg, f_true)}
            best = {}
            for name, ref in refs.items():
                lam_star, e_star = 0.0, nmse(f0)
                for lam in LAMBDAS_BEST:
                    e = nmse(f0 if lam == 0.0 else solve(reg, lam, ref))
                    if e < e_star:
                        lam_star, e_star = lam, e
                best[name] = (lam_star, e_star)
            rows.setdefault(reg_name, []).append((problem, truth, sigma, best, refs["C2"] / R0))
    for reg_name, data in rows.items():
        print(f"\n=== {reg_name} ===")
        print(f"{'problem':7} {'truth':9} {'sig':>4} | "
              f"{'C1 l*(e*)':>15} {'C2 l*(e*)':>15} {'C4 l*(e*)':>15} | {'C2 Rinit/R0':>11}")
        for problem, truth, sigma, best, anchor in data:
            cells = " ".join(f"{best[n][0]:4.2f}({best[n][1]:.3f})".rjust(15) for n in ("C1", "C2", "C4"))
            print(f"{problem:7} {truth:9} {sigma:4.2f} | {cells} | {anchor:11.3f}", flush=True)


def part3_ridge_weight():
    """the ridge-start weight: s2^2 (default) vs a lighter (1e-2) and heavier (s1^2) weight."""
    print(f"\n{'#'*84}\n# PART 3 -- the ridge-start weight (why s2^2 is the default)")
    print(f"# e0 = nmse of the l=0 solve, e* = best nmse over l under C2, anchor = R(f_init)/R(f0)\n{'#'*84}")
    for problem, truth in (("matirf", "vesicles"), ("matirf", "cell"), ("deconv", "img_001")):
        op, g, f_true, fidelity, delta = _case(problem, truth, 0.05)
        diff = DifferentialOperators(delta=delta)
        solve, R_of, nmse = _tools(op, g, f_true, fidelity, diff)
        for reg_name in ("tv", "tikhonov"):
            reg = REGULARIZATION_REGISTRY[reg_name]()
            obj0 = Objective(operator=op, g=g, data_fidelity=fidelity, regularization=reg,
                             lambda_reg=0.0, diff_ops=diff)
            weights = {"1e-2": 1e-2, "s2^2": second_eigenvalue(obj0), "s1^2": ridge_weight(obj0, "auto") ** 2}
            print(f"\n=== {problem} | {truth} | {reg_name} ===")
            print(f"{'weight':>6} {'w':>9} | {'e0':>6} {'e*':>6} {'lam*':>5} {'anchor':>7}")
            for wname, w in weights.items():
                start = ridge_start(obj0, {"init": "ridge", "lambda_rr": w})
                f0 = solve(reg, 0.0, None, start=start)
                R0, Rinit = R_of(reg, f0), R_of(reg, start)
                lam_star, e_star = 0.0, nmse(f0)
                for lam in LAMBDAS_BEST:
                    e = nmse(f0 if lam == 0.0 else solve(reg, lam, Rinit, start=start))
                    if e < e_star:
                        lam_star, e_star = lam, e
                print(f"{wname:>6} {w:9.3g} | {nmse(f0):6.3f} {e_star:6.3f} {lam_star:5.2f} "
                      f"{Rinit / R0:7.3f}", flush=True)


def main():
    torch.manual_seed(0)
    t0 = time.time()
    part1_interpretability()
    part2_reference_choice()
    part3_ridge_weight()
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
