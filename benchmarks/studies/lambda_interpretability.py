"""
λ interpretability study — the evidence behind docs/algorithms/lambda_interpretability.md (D1).

Run it from the repository root:

    env/bin/python -m benchmarks.studies.lambda_interpretability

For two regularizers (TV, Tikhonov) and two formulations of L = (1-l) D + l R:
    standard    R used as written
    calibrated  R rescaled by 1 / R(f0), f0 = the l=0 (unregularized) reconstruction
it sweeps l, solves with Adam from a ridge start, and reports
    r(l) = 1 - R(f_l)/R(f0)     (the `regularization_share` metric, benchmarks/metrics.py)
    NMSE to the truth (scale-aligned, MA-TIRF fixes f only up to a positive factor).

The question it answers: under which formulation does l read as an interpretable share?
On the cropped `vesicles` patch, the standard r(l) stays ~0 until l ~ 0.9 (l is a dead knob),
while the calibrated r(l) is graded and monotone (l reads as the share). See the note.
"""

import time

import torch

from core import DataMode, Objective
from core.metrics import optimal_scale
from pipeline import resolve_noise_model
from problems.matirf import MATIRF, MATIRF_SYNTHETIC_DIR
from solvers import Adam
from solvers.differential_operators import DifferentialOperators
from solvers.regularizers import REGULARIZATION_REGISTRY

CROP = 40          # lateral patch: keeps the study to a few seconds per regularizer
ITERS = 200
LAMBDAS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

CONFIG = {
    "input-paths": {"mode": DataMode.SYNTHETIC.value,
                    "tif": str(MATIRF_SYNTHETIC_DIR / "vesicles.TIF"),
                    "json": str(MATIRF_SYNTHETIC_DIR / "measurement_parameters.json")},
    "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": "peak"},
    "add-noise": {"gaussian": 0.02, "seed": 1},
    "algo-params": {},
}


class _CalibratedObjective(Objective):
    """L = (1-l) D + l * (R / R_ref): R rescaled so the two terms are comparable at f0."""

    def __init__(self, *args, reference, **kwargs):
        super().__init__(*args, **kwargs)
        self._inv = 1.0 / reference

    def reg_term(self, f):
        return super().reg_term(f) * self._inv

    def prox_reg(self, f, tau=1.0):
        return super().prox_reg(f, tau * self._inv)


def _solve(op, g, fidelity, diff, reg, lam, reference):
    kind = dict(operator=op, g=g, data_fidelity=fidelity, regularization=reg,
                lambda_reg=lam, diff_ops=diff)
    obj = (_CalibratedObjective(**kind, reference=reference) if reference is not None
           else Objective(**kind))
    start = Adam().initial_guess(obj, {"init": "ridge", "lambda_rr": 1e-2})
    return Adam().solve(obj, start, {"max_iter": ITERS})


def main():
    torch.manual_seed(0)
    prepared = MATIRF.prepare(CONFIG)
    op = prepared.operator
    # resolve the noise model on the FULL measurement, then crop (fidelity holds (a, b), not g)
    fidelity, _ = resolve_noise_model(prepared, CONFIG)
    c = CROP
    g = prepared.g[..., :c, :c].contiguous()               # H acts per column in z: same H,
    f_true = prepared.f_true[..., :c, :c].contiguous()      # fewer columns
    diff = DifferentialOperators(delta=float(op.estimate_anisotropy_ratio()))

    def nmse(f):
        alpha = optimal_scale(f, f_true)
        return float(((alpha * f - f_true).norm() / f_true.norm()) ** 2)

    print(f"crop {c}x{c} | f_true peak {float(f_true.max()):.4g} | "
          f"g peak {float(g.max()):.4g} | delta {diff.delta:.3g}", flush=True)

    for reg_name in ("tv", "tikhonov"):
        reg = REGULARIZATION_REGISTRY[reg_name]()
        started = time.time()
        f0 = _solve(op, g, fidelity, diff, reg, 0.0, None)   # the calibration reference
        R0 = float(reg.loss(f0, diff))
        print(f"\n=== {reg_name} | R(f0)={R0:.4g} | nmse(f0)={nmse(f0):.3f} ===")
        print(f"{'lambda':>7} | {'r std':>7} {'nmse std':>9} | {'r cal':>7} {'nmse cal':>9}")
        for lam in LAMBDAS:
            fs = _solve(op, g, fidelity, diff, reg, lam, None)
            fc = _solve(op, g, fidelity, diff, reg, lam, R0) if lam > 0 else f0
            rs = 1 - float(reg.loss(fs, diff)) / R0 if R0 > 0 else float("nan")
            rc = 1 - float(reg.loss(fc, diff)) / R0 if R0 > 0 else float("nan")
            print(f"{lam:7.2f} | {rs:7.3f} {nmse(fs):9.3f} | {rc:7.3f} {nmse(fc):9.3f}",
                  flush=True)
        print(f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
