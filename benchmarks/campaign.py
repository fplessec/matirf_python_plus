"""
The benchmark campaign — which runs make up the report, at full resolution.

    phase A (atlas)   every method on every structure and noise level.
    phase B (lambda)  a lambda sweep for the reg-based reference (ADAM, TV & Tikhonov),
                      so `regularization_share` r(lambda) can be plotted.
    phase C (v1/v2)   NOT extra runs: it reads the v1/v2 pairs already in phase A.

`specs()` returns the RunSpec list (phase A + phase B, de-duplicated by run_id) to hand to
`run_campaign`. Configs are full and explicit; the ridge s2^2 start is the solver default,
so it is not set here. Noise is oracle (D uses the exact injected sigma^2).

    from benchmarks.campaign import specs
    from benchmarks.runner import run_campaign
    run_campaign(specs(), "benchmarks/results/main")
"""
from core import DataMode
from benchmarks.runner import RunSpec
from problems.deconv import DECONV_MEASUREMENTS_DIR
from problems.matirf import MATIRF_SYNTHETIC_DIR

# ── the matrix (full resolution) ──────────────────────────────────────────────

## per-solver iteration budget, per family. Mostly the authors' tuned defaults; PPXA is
## capped at 500 (from 2000) because at full resolution it costs ~150 s/run on MA-TIRF and
## still under-converges there (nmse ~0.6) — the report notes its slow convergence.
SOLVER_ITERS = {
    "ADAM": {"max_iter": 1000}, "PPXA": {"max_iter": 500},
    "ADMM": {"iter": 20}, "ADMMv2": {"iter": 100},
    "PNP": {"iter": 5}, "PNPv2": {"iter": 16},
    "ADMM-PnP": {"iter": 20}, "ADMM-PnPv2": {"iter": 50},
    "MCMC": {"max_iter": 200}, "MCMCv2": {"max_iter": 300},
}
ALL_SOLVERS = list(SOLVER_ITERS)
DECONV_SOLVERS = ["ADAM", "PPXA", "MCMC"]        # the plan: deconv keeps the general solvers
PNP_SOLVERS = {"PNP", "PNPv2", "ADMM-PnP", "ADMM-PnPv2"}
MCMC_SOLVERS = {"MCMC", "MCMCv2"}
DENOISER_SOLVERS = PNP_SOLVERS | MCMC_SOLVERS
REG_SOLVERS = {"ADAM", "PPXA"}                   # take reg + lambda_reg on the shared objective

## anisotropic-aware, never NL-Ridge / DCT. TV Bregman is the robust denoiser for the MCMC
## chain (the MCMC study: Bilateral drifts); PnP keeps the anisotropic Bilateral.
PNP_DENOISER = "Bilateral"
MCMC_DENOISER = "TV Bregman"
ATLAS_REG, ATLAS_LAMBDA = "tv", 0.1              # for the reg-based solvers in the atlas
NOISES = [0.02, 0.05]                            # sigma as a fraction of the peak (low, moderate)
MATIRF_TRUTHS = ["vesicles", "fibres", "cell", "cell_fibres_vesicles"]
DECONV_TRUTHS = ["img_001", "img_002", "img_003", "img_004"]

## phase B: the lambda sweep (ADAM, the reg-based reference), on one truth per problem
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75]
LAMBDA_REGS = ["tv", "tikhonov"]
LAMBDA_MATIRF_TRUTH, LAMBDA_DECONV_TRUTH, LAMBDA_SIGMA = "fibres", "img_001", 0.05

JSON_MATIRF = str(MATIRF_SYNTHETIC_DIR / "measurement_parameters.json")
JSON_DECONV = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")

# ── config builders ───────────────────────────────────────────────────────────

def _algo_params(solver, *, reg=None, lam=None):
    params = dict(SOLVER_ITERS[solver])
    if solver in MCMC_SOLVERS:
        params["denoiser"] = MCMC_DENOISER
    elif solver in PNP_SOLVERS:
        params["denoiser"] = PNP_DENOISER
    if solver in REG_SOLVERS:
        params["reg"] = reg if reg is not None else ATLAS_REG
        params["lambda_reg"] = ATLAS_LAMBDA if lam is None else lam
    return params


def matirf_config(truth, solver, sigma, *, reg=None, lam=None):
    return {
        "algorithm": solver,
        "input-paths": {"mode": DataMode.SYNTHETIC.value,
                        "tif": str(MATIRF_SYNTHETIC_DIR / f"{truth}.TIF"), "json": JSON_MATIRF},
        "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": "peak"},
        "add-noise": {"gaussian_noise": True, "sigma": sigma, "seed": 1},
        "noise-model": {"parameters": "oracle"},
        "algo-params": _algo_params(solver, reg=reg, lam=lam),
    }


def deconv_config(truth, solver, sigma, *, reg=None, lam=None):
    return {
        "algorithm": solver,
        "input-paths": {"mode": DataMode.SYNTHETIC.value,
                        "png": str(DECONV_MEASUREMENTS_DIR / f"{truth}.png"),
                        "json": JSON_DECONV, "normalization": "peak"},
        "add-noise": {"gaussian_noise": True, "sigma": sigma, "seed": 1},
        "noise-model": {"parameters": "oracle"},
        "algo-params": _algo_params(solver, reg=reg, lam=lam),
    }

# ── the phases ────────────────────────────────────────────────────────────────

def atlas_specs():
    """Phase A: every method on every structure and noise level."""
    out = []
    for sigma in NOISES:
        for truth in MATIRF_TRUTHS:
            for solver in ALL_SOLVERS:
                out.append(RunSpec("matirf", matirf_config(truth, solver, sigma),
                                   tags={"phase": "A", "problem": "matirf", "solver": solver,
                                         "truth": truth, "sigma": sigma}))
        for truth in DECONV_TRUTHS:
            for solver in DECONV_SOLVERS:
                out.append(RunSpec("deconv", deconv_config(truth, solver, sigma),
                                   tags={"phase": "A", "problem": "deconv", "solver": solver,
                                         "truth": truth, "sigma": sigma}))
    return out


def lambda_specs():
    """Phase B: a lambda sweep for ADAM (TV & Tikhonov), one truth per problem."""
    out = []
    for reg in LAMBDA_REGS:
        for lam in LAMBDAS:
            out.append(RunSpec("matirf",
                               matirf_config(LAMBDA_MATIRF_TRUTH, "ADAM", LAMBDA_SIGMA, reg=reg, lam=lam),
                               tags={"phase": "B", "problem": "matirf", "solver": "ADAM",
                                     "truth": LAMBDA_MATIRF_TRUTH, "sigma": LAMBDA_SIGMA,
                                     "reg": reg, "lambda": lam}))
            out.append(RunSpec("deconv",
                               deconv_config(LAMBDA_DECONV_TRUTH, "ADAM", LAMBDA_SIGMA, reg=reg, lam=lam),
                               tags={"phase": "B", "problem": "deconv", "solver": "ADAM",
                                     "truth": LAMBDA_DECONV_TRUTH, "sigma": LAMBDA_SIGMA,
                                     "reg": reg, "lambda": lam}))
    return out


def specs():
    """Phase A + phase B, de-duplicated by run_id (an atlas run and a lambda run can coincide)."""
    seen, out = set(), []
    for spec in atlas_specs() + lambda_specs():
        if spec.run_id not in seen:
            seen.add(spec.run_id)
            out.append(spec)
    return out


def summary():
    """A human count of the campaign, by phase and problem — printed before launch."""
    all_specs = specs()
    by = {}
    for spec in all_specs:
        key = (spec.tags["phase"], spec.tags["problem"])
        by[key] = by.get(key, 0) + 1
    lines = [f"Campaign: {len(all_specs)} runs (full resolution)"]
    for (phase, problem), n in sorted(by.items()):
        lines.append(f"  phase {phase} · {problem:7} {n:4d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
