"""
The two metrics that do NOT fit core.metrics' (f, f_true) contract.

Everything that compares a reconstruction to a ground truth — nmse, psnr, ssim, and the MA-TIRF
depth metrics (Depth Error, Stack Recovery) — lives in `core.metrics` and is computed by the
pipeline (the benchmark curates the subset it keeps, see benchmarks/runner.py CURATED_3D/2D).
Only these two remain here, because neither compares to a truth:

    chi2_ratio             fitting the data or the noise? mean residual^2 / noise variance:
                           ~1 the right fit, << 1 over-fits the noise, >> 1 over-regularizes.
                           Needs no truth, so it is the one quality number on the real esoubies
                           data. (a, b) are the noise model's own variance parameters.
    regularization_share   how regularized? r = 1 - R(f_lambda) / R(f_0), f_0 the unregularized
                           reconstruction. Needs the paired lambda=0 run, so the analysis
                           computes it across a sweep — not per run. What "lambda_reg = 0.1 means
                           10 % regularized" is tested against.
"""

import torch


def chi2_ratio(f: torch.Tensor, operator, g: torch.Tensor, a: float, b: float) -> float:
    """mean (Hf - g)^2 / Var, Var = a Hf + b — the noise model's own variance (a, b)."""
    Hf = operator.apply(f.detach())
    variance = (a * Hf.clamp(min=0) + b).clamp(min=1e-12)
    ## Hf is only defined up to the scale of f (MA-TIRF): align it to g first
    alpha = float((Hf * g).sum() / (Hf * Hf).sum().clamp(min=1e-300))
    return float(((alpha * Hf - g) ** 2 / variance).mean())


def regularization_share(R_lambda: float, R_zero: float) -> float:
    """1 - R(f_lambda) / R(f_0): how much of the unregularized irregularity was removed."""
    return float(1.0 - R_lambda / R_zero) if R_zero > 0 else float("nan")
