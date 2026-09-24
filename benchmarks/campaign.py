"""
Phase A — the parameter atlas, derived from docs/algorithms/ (the a priori analysis).

For each solver we sweep ONLY the axes the theory says move the *solution* (foundation §3.4);
the convergence knobs are fixed at the values the notes justify. Full resolution, physical
operator (`normalize = False`), g peak-normalized. Datasets: two synthetic MA-TIRF truths
(cell_fibres_vesicles, vesicles) at three noise levels (none, gaussian 0.02, gaussian 0.05),
the real esoubies measurement (native noise, no truth), and one deconvolution image (img_001)
at the three noise levels.

Solvers: MA-TIRF gets all six; deconvolution gets the general ones (Adam, PPXA, MCMC).
PPXA is CONFIRMATION-ONLY: Adam and PPXA minimize the same objective, so the reg×λ atlas is
Adam's; PPXA runs a handful of points to verify Adam=PPXA and to test its γ range.

    from benchmarks.campaign import specs
    from benchmarks.runner import run_campaign
    run_campaign(specs(), "benchmarks/results/main")
"""
from core import DataMode
from benchmarks.runner import RunSpec
from problems.deconv import DECONV_MEASUREMENTS_DIR
from problems.matirf import MATIRF_MEASUREMENTS_DIR, MATIRF_SYNTHETIC_DIR

# ── datasets and noise ────────────────────────────────────────────────────────

MATIRF_SYNTHETIC = ["cell_fibres_vesicles", "vesicles"]     # have a ground truth
MATIRF_REAL = "esoubies"                                     # real data, no truth
DECONV_IMAGE = "img_001"
NOISES = [0.0, 0.02, 0.05]                                   # none, then two Gaussian levels
JSON_MATIRF = str(MATIRF_SYNTHETIC_DIR / "measurement_parameters.json")
JSON_ESOUBIES = str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")
JSON_DECONV = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")

# ── the parameter grids (from the notes) ──────────────────────────────────────

LAMBDAS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]                  # Adam/PPXA λ (calibrated share)
SHV_RHO = 0.6                                                # + TV and the no-prior case (λ = 0)
ADAM_FIXED = {"max_iter": 3000, "K": 10, "EPS": 1e-8}        # EPS stops before the budget

ADMM_KAPPA = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]                  # threshold_ratio (log band)
ADMM_MU = [0.3, 0.5, 1.0]

DENOISERS = ["Gaussian", "Bilateral"]                        # + the no-denoiser case
PNP_SIGMA = [10.0, 25.0, 50.0]                               # 0-255 scale
PNP_LAMBDA_KZ = [1e-3, 1e-2, 1e-1]                           # final data weight (≈ s4²..s3²)
ADMMPNP_RHO = [1e-3, 1e-2, 1e-1]
MCMC_SIGMA = [0.02, 0.03, 0.05]                              # fraction of the peak

# ── config builders (physical operator, g peak-normalized) ────────────────────

def _noise_sections(noise):
    """(add-noise, noise-model) for a synthetic run: none → estimated (≈ unscaled), else oracle."""
    if noise == 0.0:
        return {}, {}                                        # no added noise; noise estimated
    return {"gaussian_noise": True, "sigma": noise, "seed": 1}, {"parameters": "oracle"}


def matirf_synth_config(truth, solver, noise, algo_params):
    add_noise, noise_model = _noise_sections(noise)
    return {"algorithm": solver,
            "input-paths": {"mode": DataMode.SYNTHETIC.value,
                            "tif": str(MATIRF_SYNTHETIC_DIR / f"{truth}.TIF"),
                            "json": JSON_MATIRF, "normalization": "peak"},
            "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": False},
            "add-noise": add_noise, "noise-model": noise_model, "algo-params": algo_params}


def matirf_real_config(solver, algo_params):
    """esoubies: real data (native noise, no truth); noise estimated from g."""
    return {"algorithm": solver,
            "input-paths": {"mode": DataMode.REAL.value,
                            "tif": str(MATIRF_MEASUREMENTS_DIR / "esoubies.TIF"),
                            "json": JSON_ESOUBIES, "normalization": "peak"},
            "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": False},
            "add-noise": {}, "noise-model": {}, "algo-params": algo_params}


def deconv_config(solver, noise, algo_params):
    add_noise, noise_model = _noise_sections(noise)
    return {"algorithm": solver,
            "input-paths": {"mode": DataMode.SYNTHETIC.value,
                            "png": str(DECONV_MEASUREMENTS_DIR / f"{DECONV_IMAGE}.png"),
                            "json": JSON_DECONV, "normalization": "peak"},
            "add-noise": add_noise, "noise-model": noise_model, "algo-params": algo_params}

# ── per-solver algo-params (the swept axis + the fixed knobs) ──────────────────

def _reg_grid():
    """(tag, algo-params) for Adam/PPXA: the no-prior point, then TV and SHV over λ."""
    yield ({"reg": "none", "lambda": 0.0}, {"lambda_reg": 0.0, **ADAM_FIXED})
    for lam in LAMBDAS:
        yield ({"reg": "tv", "lambda": lam}, {"reg": "tv", "lambda_reg": lam, **ADAM_FIXED})
        yield ({"reg": "shv", "lambda": lam},
               {"reg": "shv", "rho": SHV_RHO, "lambda_reg": lam, **ADAM_FIXED})


def _denoiser_grid(strength_name, strengths):
    """(tag, params) for a PnP-family solver: the no-denoiser point, then 2 denoisers × σ × strength."""
    yield ({"denoiser": "None"}, {"denoiser": "None"})
    for denoiser in DENOISERS:
        for sigma in PNP_SIGMA:
            for strength in strengths:
                yield ({"denoiser": denoiser, "sigma": sigma, strength_name: strength},
                       {"denoiser": denoiser, "sigma": sigma, strength_name: strength})


def _mcmc_grid():
    yield ({"denoiser": "None"}, {"denoiser": "None"})
    for denoiser in DENOISERS:
        for sigma in MCMC_SIGMA:
            yield ({"denoiser": denoiser, "sigma": sigma}, {"denoiser": denoiser, "sigma": sigma})


## every solver's swept axis and its fixed knobs, as (tag_extra, algo-params) generators
def _solver_specs(solver):
    if solver == "ADAM":
        yield from _reg_grid()
    elif solver == "ADMM":
        for kappa in ADMM_KAPPA:
            for mu in ADMM_MU:
                yield ({"kappa": kappa, "mu": mu},
                       {"iter": 200, "threshold_ratio": kappa, "mu": mu})
    elif solver == "PNP":
        for tag, p in _denoiser_grid("lambda_kz", PNP_LAMBDA_KZ):
            yield (tag, {"iter": 16, **p})
    elif solver == "ADMM-PnP":
        for tag, p in _denoiser_grid("rho", ADMMPNP_RHO):
            yield (tag, {"iter": 50, **p})
    elif solver == "MCMC":
        for tag, p in _mcmc_grid():
            yield (tag, {"max_iter": 300, **p})


MATIRF_SOLVERS = ["ADAM", "ADMM", "PNP", "ADMM-PnP", "MCMC"]     # PPXA handled separately
DECONV_SOLVERS = ["ADAM", "MCMC"]                                # + PPXA confirmation

# ── PPXA confirmation-only (Adam = PPXA on the same objective; + a γ range check) ─

def _ppxa_confirmation():
    """~10 runs: verify Adam=PPXA on TV/SHV at λ=0.1, and sweep γ once."""
    base = {"max_iter": 3000, "K": 10, "EPS": 1e-8, "lambda_relax": 1.5, "gamma": 0.01}
    out = []
    for truth in MATIRF_SYNTHETIC:                       # confirm on both truths, TV + SHV, λ=0.1, σ=0.02
        for reg, extra in (("tv", {}), ("shv", {"rho": SHV_RHO})):
            ap = {"reg": reg, "lambda_reg": 0.1, **extra, **base}
            out.append(RunSpec("matirf", matirf_synth_config(truth, "PPXA", 0.02, ap),
                               tags={"phase": "A", "problem": "matirf", "solver": "PPXA",
                                     "dataset": truth, "noise": 0.02, "reg": reg, "lambda": 0.1,
                                     "role": "confirm"}))
    for gamma in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:         # γ range check on TV, vesicles
        ap = {"reg": "tv", "lambda_reg": 0.1, **base, "gamma": gamma}
        out.append(RunSpec("matirf", matirf_synth_config("vesicles", "PPXA", 0.02, ap),
                           tags={"phase": "A", "problem": "matirf", "solver": "PPXA",
                                 "dataset": "vesicles", "noise": 0.02, "gamma": gamma,
                                 "role": "gamma"}))
    ap = {"reg": "tv", "lambda_reg": 0.1, **base}        # deconv confirm
    out.append(RunSpec("deconv", deconv_config("PPXA", 0.02, ap),
                       tags={"phase": "A", "problem": "deconv", "solver": "PPXA",
                             "dataset": DECONV_IMAGE, "noise": 0.02, "reg": "tv", "lambda": 0.1,
                             "role": "confirm"}))
    return out

# ── the campaign ──────────────────────────────────────────────────────────────

def _matirf_specs():
    out = []
    for solver in MATIRF_SOLVERS:
        for tag_extra, ap in _solver_specs(solver):
            for truth in MATIRF_SYNTHETIC:
                for noise in NOISES:
                    out.append(RunSpec("matirf", matirf_synth_config(truth, solver, noise, ap),
                                       tags={"phase": "A", "problem": "matirf", "solver": solver,
                                             "dataset": truth, "noise": noise, **tag_extra}))
            out.append(RunSpec("matirf", matirf_real_config(solver, ap),
                               tags={"phase": "A", "problem": "matirf", "solver": solver,
                                     "dataset": MATIRF_REAL, "noise": "native", **tag_extra}))
    return out


def _deconv_specs():
    out = []
    for solver in DECONV_SOLVERS:
        for tag_extra, ap in _solver_specs(solver):
            for noise in NOISES:
                out.append(RunSpec("deconv", deconv_config(solver, noise, ap),
                                   tags={"phase": "A", "problem": "deconv", "solver": solver,
                                         "dataset": DECONV_IMAGE, "noise": noise, **tag_extra}))
    return out


def specs():
    """All Phase-A runs, de-duplicated by run_id."""
    seen, out = set(), []
    for spec in _matirf_specs() + _deconv_specs() + _ppxa_confirmation():
        if spec.run_id not in seen:
            seen.add(spec.run_id)
            out.append(spec)
    return out


def summary():
    all_specs = specs()
    by = {}
    for spec in all_specs:
        key = (spec.tags["problem"], spec.tags["solver"])
        by[key] = by.get(key, 0) + 1
    lines = [f"Phase A: {len(all_specs)} runs (full resolution)"]
    for (problem, solver), n in sorted(by.items()):
        lines.append(f"  {problem:7} {solver:12} {n:4d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
