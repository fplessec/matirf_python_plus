"""
The benchmark — a CURATED set of runs, one clean campaign.

Not a parameter sweep. The a priori analysis (docs/algorithms/) and the earlier campaigns
already told us, for each algorithm, where it wins, its sensible default, and where it fails
(depth collapse / too slow / out of memory). So this file lists exactly the runs the report
needs and nothing more: for each algorithm, its STANDARD (default) config and its OPTIMAL
(best-known) config, on the reference truth, across the three noise levels — plus one real-data
point. Every config carries a `role` tag (standard / alt / confirm / real) so the analysis can
build each algorithm's "standard vs optimal" table.

Reference truth: `cell_fibres_vesicles` (a composite of filaments + vesicles — it exercises both
the sparsity and the denoiser priors). Deconvolution (`img_001`) is the well-posed companion for
the general solvers. `esoubies` is the real measurement (no truth) for the real-data illustration.

COMPLETE CONFIGS, for reproducibility: every parameter a solver reads is written explicitly,
including the ones held fixed for the whole campaign (max_iter, K, EPS, lr, the ridge start and
its weight, the anisotropy delta, ...). Nothing is left to an implicit default. The two
geometry-derived weights are pinned to named constants below (their estimated values for this
fixed optics/grid). The only parameters not written are the genuinely automatic, noise-dependent
ones (PnP's `sigma_final`, and `lambda_rr` on the 2D deconvolution companion, whose value comes
from the PSF) — they are documented where they occur.

    from benchmarks.campaign import specs
    from benchmarks.runner import run_campaign
    run_campaign(specs(), "benchmarks/results/main")
"""
from benchmarks.runner import RunSpec
from core import DataMode
from problems.deconv import DECONV_MEASUREMENTS_DIR
from problems.matirf import MATIRF_MEASUREMENTS_DIR, MATIRF_SYNTHETIC_DIR

# ── datasets ──────────────────────────────────────────────────────────────────

## three primitive structures — the denoiser winner depends on the object, so the benchmark
## sweeps them: point-like (vesicles), filaments (fibres), continuous membrane (cell).
TRUTHS = ["vesicles", "fibres", "cell"]
REF_TRUTH = "fibres"                     # the reference for the reference-condition studies (lambda-shape)
DECONV_IMAGE = "img_001"                # the 2D deconvolution companion
NOISES = [0.0, 0.02, 0.05]              # none, then two Gaussian levels
JSON_MATIRF = str(MATIRF_SYNTHETIC_DIR / "measurement_parameters.json")
JSON_ESOUBIES = str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")
JSON_DECONV = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")

# ── geometry-derived constants, pinned explicitly for a self-contained config ──

## Anisotropy delta = dz/dxy for this fixed optics + grid (nz=50 over 0-300 nm, NA 1.33,
## lambda 491 nm, n 1.34): dz=6 nm, dxy~84 nm. It is the value the operator would estimate; we
## write it into every MA-TIRF config so a reader needs nothing outside the file to reproduce it.
MATIRF_DELTA = 0.0714046275583453
## Ridge-start weight for the MAP/denoiser solvers whose start is (H^TH + lambda_rr I)^-1H^Tg with
## lambda_rr = s2^2 (second eigenvalue of H^TH; same for the synthetic and esoubies geometries).
MATIRF_RIDGE = 36.24
## MCMC's ridge weight is its own automatic value s1 = sqrt||H^TH|| (the largest singular value),
## pinned here so the config is explicit.
MATIRF_S1 = 38.863
DECONV_DELTA = 1.0                       # 2D deconvolution: isotropic grid (dz = dxy)
SHV_RHO = 0.6                            # SHV sparse/smooth balance (docs adam_ppxa §1.3)

# ── fixed convergence knobs (justified in docs/algorithms/) ───────────────────

## Adam / PPXA: large-enough first steps (the scheduler halves on overshoot), the ridge warm
## start, a strict plateau stop with a big budget so EPS — not the budget — ends the run.
ADAM_FIXED = {"max_iter": 5000, "K": 10, "EPS": 1e-8, "lr": 0.1, "init": "ridge"}
PPXA_FIXED = {"max_iter": 5000, "K": 10, "EPS": 1e-8, "lambda_relax": 1.5, "gamma": 0.01,
              "init": "ridge"}          # gamma inside [1/s1^2, 1/s2^2] (docs adam_ppxa §3.2)

# ── config builders (physical operator, g peak-normalized, normalize=False) ───

def _noise_sections(noise):
    """(add-noise, noise-model) for a synthetic run: none -> estimated, else the oracle level."""
    if noise == 0.0:
        return {}, {}
    return {"gaussian_noise": True, "sigma": noise, "seed": 1}, {"parameters": "oracle"}


def matirf_config(truth, solver, noise, algo_params):
    add_noise, noise_model = _noise_sections(noise)
    return {"algorithm": solver,
            "input-paths": {"mode": DataMode.SYNTHETIC.value,
                            "tif": str(MATIRF_SYNTHETIC_DIR / f"{truth}.TIF"),
                            "json": JSON_MATIRF, "normalization": "peak"},
            "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": False},
            "add-noise": add_noise, "noise-model": noise_model, "algo-params": algo_params}


def esoubies_config(solver, algo_params):
    """The real esoubies measurement: native noise, no truth (noise estimated from g)."""
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

# ── per-solver COMPLETE algo-params (every parameter the solver reads) ─────────

def _adam_matirf(reg, lam):
    """Adam on MA-TIRF: prior `reg` at share `lam`, ridge s2^2 start, pinned anisotropy."""
    params = {"reg": reg, "lambda_reg": lam, **ADAM_FIXED,
              "lambda_rr": MATIRF_RIDGE, "delta": MATIRF_DELTA}
    if reg == "shv":
        params["rho"] = SHV_RHO
    return params


def _adam_deconv(reg, lam):
    ## deconv `lambda_rr` is automatic (the PSF's s1, not a fixed choice) — left out on purpose.
    params = {"reg": reg, "lambda_reg": lam, **ADAM_FIXED, "delta": DECONV_DELTA}
    if reg == "shv":
        params["rho"] = SHV_RHO
    return params


def _ppxa_matirf(reg, lam):
    return {"reg": reg, "lambda_reg": lam, **PPXA_FIXED,
            "lambda_rr": MATIRF_RIDGE, "delta": MATIRF_DELTA}


def _admm(kappa):
    ## ADMM reads only these three (its start is its own data step, independent of the ridge).
    return {"iter": 200, "mu": 0.5, "threshold_ratio": kappa}


def _pnp_matirf(denoiser, sigma, lambda_kz, n_iter):
    ## sigma_final is automatic (tied to the measurement's noise) — the one non-fixed knob.
    ## lambda_kz = alpha_final, the eigenvalue threshold s_i^2 >> lambda_kz -> data, else denoiser;
    ## on MA-TIRF's spectrum lambda_kz ~ 12 hands the whole null space to the prior (the winner).
    return {"iter": n_iter, "denoiser": denoiser, "sigma": sigma, "lambda_kz": lambda_kz,
            "init": "ridge", "lambda_rr": MATIRF_RIDGE, "delta": MATIRF_DELTA}


def _admm_pnp_matirf(denoiser, sigma, rho, n_iter=50):
    ## prior strength ~ rho*sigma^2; the dual variable fills the null space, so LOW rho wins
    ## (opposite of PnP's lambda_kz). Gaussian is the winner here; TV Bregman collapses.
    return {"iter": n_iter, "denoiser": denoiser, "sigma": sigma, "rho": rho,
            "forced_pos": True, "init": "ridge", "lambda_rr": MATIRF_RIDGE, "delta": MATIRF_DELTA}


def _mcmc_matirf(denoiser, sigma, lambda_rr, n_iter):
    ## lambda_rr is the data-step ridge: its useful window is [s3^2, s2^2] ~ [0.3, 36]; the old
    ## default s1=38.9 sits ABOVE it (pull too weak) and crippled MCMC. lambda_rr ~ 5 is the winner.
    return {"max_iter": n_iter, "K": 50, "denoiser": denoiser, "sigma": sigma,
            "temperature": 1.0, "init": "ridge", "lambda_rr": lambda_rr}


def _mcmc_deconv(denoiser, sigma):
    ## deconv lambda_rr automatic (PSF-derived s1) — left out on purpose.
    return {"max_iter": 300, "K": 50, "denoiser": denoiser, "sigma": sigma,
            "temperature": 1.0, "init": "ridge"}


def _spec(problem, config, solver, dataset, noise, role, ptags):
    """A RunSpec with report labels: the run's identity (solver/dataset/noise/role) + its
    swept parameters (ptags), for the analysis tables."""
    return RunSpec(problem, config,
                   tags={"problem": problem, "solver": solver, "dataset": dataset,
                         "noise": noise, "role": role, **ptags})

# ── the curated campaign ──────────────────────────────────────────────────────

## Adam's two priors: sparsity (L1) vs smoothness (Tikhonov, the L2 of the gradient — a degree-2
## prior, ~50x weaker at f~0.02). L1 is the winner; Tikhonov is the sensible smooth default.
ADAM_LAMBDAS = (0.05, 0.1, 0.2, 0.3, 0.5)          # the shape sweep (at the reference noise)


## denoiser sweeps, each denoiser at its good regime (tuned on the composite; the benchmark
## reveals which one wins per structure). role="denoiser" — analysis picks the best per cell.
PNP_DENOISERS = [("Gaussian", 25.0, 12.0), ("TV Bregman", 40.0, 12.0), ("DCT", 18.0, 6.0)]      # (den, sigma, lambda_kz), iter=40
ADMMPNP_DENOISERS = [("Gaussian", 50.0, 0.03), ("TV Bregman", 40.0, 4.0), ("DCT", 22.0, 1.0)]   # (den, sigma, rho), iter=50
MCMC_DENOISERS = [("TV Bregman", 0.01, 5.0), ("Gaussian", 0.01, 5.0), ("DCT", 0.01, 5.0)]       # (den, sigma, lambda_rr), iter=300


def _adam():
    """Adam — sparsity (L1) vs smoothness (Tikhonov) on each of the three structures: standard
    Tikhonov, optimal L1 (the prior winner may itself depend on the object). The lambda-shape (why L1
    has an optimum) is drawn on the reference structure at noise 0.02."""
    out = []
    for truth in TRUTHS:
        for noise in NOISES:
            out.append(_spec("matirf", matirf_config(truth, "ADAM", noise, _adam_matirf("tikhonov", 0.1)),
                             "ADAM", truth, noise, "standard", {"reg": "tikhonov", "lambda": 0.1}))
            out.append(_spec("matirf", matirf_config(truth, "ADAM", noise, _adam_matirf("none", 0.0)),
                             "ADAM", truth, noise, "alt", {"reg": "none", "lambda": 0.0}))
            out.append(_spec("matirf", matirf_config(truth, "ADAM", noise, _adam_matirf("l1", 0.2)),
                             "ADAM", truth, noise, "optimal", {"reg": "l1", "lambda": 0.2}))
    for reg in ("l1", "tikhonov"):                             # the lambda-shape at the reference truth
        for lam in ADAM_LAMBDAS:
            out.append(_spec("matirf", matirf_config(REF_TRUTH, "ADAM", 0.02, _adam_matirf(reg, lam)),
                             "ADAM", REF_TRUTH, 0.02, "shape", {"reg": reg, "lambda": lam}))
    for noise in NOISES:                                       # deconvolution companion
        out.append(_spec("deconv", deconv_config("ADAM", noise, _adam_deconv("tikhonov", 0.1)),
                         "ADAM", DECONV_IMAGE, noise, "standard", {"reg": "tikhonov", "lambda": 0.1}))
        out.append(_spec("deconv", deconv_config("ADAM", noise, _adam_deconv("none", 0.0)),
                         "ADAM", DECONV_IMAGE, noise, "alt", {"reg": "none", "lambda": 0.0}))
        out.append(_spec("deconv", deconv_config("ADAM", noise, _adam_deconv("l1", 0.2)),
                         "ADAM", DECONV_IMAGE, noise, "optimal", {"reg": "l1", "lambda": 0.2}))
    return out


def _ppxa():
    """PPXA — one confirmation point (Adam = PPXA on the same objective), on L1 (both prox it)."""
    return [_spec("matirf", matirf_config(REF_TRUTH, "PPXA", 0.02, _ppxa_matirf("l1", 0.2)),
                  "PPXA", REF_TRUTH, 0.02, "confirm", {"reg": "l1", "lambda": 0.2})]


def _admm_specs():
    """ADMM — soft-threshold sparsity, on each structure. Standard kappa=0.1; sparser kappa=0.01 alt."""
    out = []
    for truth in TRUTHS:
        for noise in NOISES:
            for kappa, role in ((0.1, "standard"), (0.01, "alt")):
                out.append(_spec("matirf", matirf_config(truth, "ADMM", noise, _admm(kappa)),
                                 "ADMM", truth, noise, role, {"kappa": kappa, "mu": 0.5}))
    return out


def _pnp():
    """PnP-HQS — annealing-schedule denoiser prior (iter=40, lambda_kz=12: hands the null space to the
    prior). Standard = the DPIR default (Gaussian sigma=25, lambda_kz=0.23, iter=16). Sweeps Gaussian / TV
    Bregman / DCT per structure — the winner (TV Bregman on the composite) may depend on the object."""
    out = []
    for truth in TRUTHS:
        for noise in NOISES:
            out.append(_spec("matirf", matirf_config(truth, "PNP", noise, _pnp_matirf("Gaussian", 25.0, 0.23, 16)),
                             "PNP", truth, noise, "standard",
                             {"denoiser": "Gaussian", "sigma": 25, "lambda_kz": 0.23, "iter": 16}))
            for den, sigma, lkz in PNP_DENOISERS:
                out.append(_spec("matirf", matirf_config(truth, "PNP", noise, _pnp_matirf(den, sigma, lkz, 40)),
                                 "PNP", truth, noise, "denoiser",
                                 {"denoiser": den, "sigma": sigma, "lambda_kz": lkz, "iter": 40}))
    return out


def _admm_pnp():
    """ADMM-PnP — dual-variable denoiser prior (iter=50). Standard = Gaussian sigma=25, rho=0.1. Sweeps
    Gaussian / TV Bregman / DCT per structure (Gaussian wins on the composite; the dual makes it
    prefer a moderate rho*sigma^2 — opposite of PnP). + the real esoubies point with the best config."""
    out = []
    for truth in TRUTHS:
        for noise in NOISES:
            out.append(_spec("matirf", matirf_config(truth, "ADMM-PnP", noise, _admm_pnp_matirf("Gaussian", 25.0, 0.1)),
                             "ADMM-PnP", truth, noise, "standard",
                             {"denoiser": "Gaussian", "sigma": 25, "rho": 0.1}))
            for den, sigma, rho in ADMMPNP_DENOISERS:
                out.append(_spec("matirf", matirf_config(truth, "ADMM-PnP", noise, _admm_pnp_matirf(den, sigma, rho)),
                                 "ADMM-PnP", truth, noise, "denoiser",
                                 {"denoiser": den, "sigma": sigma, "rho": rho}))
    return out


def _esoubies():
    """The real esoubies stack (no truth): EVERY solver reconstructs it at its good config.
    Judged by chi2_ratio and by eye; also exposes each solver's runtime and memory on a full
    real volume (the PnP family is the memory-hungry one here)."""
    return [
        _spec("matirf", esoubies_config("ADAM", _adam_matirf("l1", 0.2)),
              "ADAM", "esoubies", "native", "real", {"reg": "l1", "lambda": 0.2}),
        _spec("matirf", esoubies_config("PPXA", _ppxa_matirf("l1", 0.2)),
              "PPXA", "esoubies", "native", "real", {"reg": "l1", "lambda": 0.2}),
        _spec("matirf", esoubies_config("ADMM", _admm(0.1)),
              "ADMM", "esoubies", "native", "real", {"kappa": 0.1, "mu": 0.5}),
        _spec("matirf", esoubies_config("PNP", _pnp_matirf("TV Bregman", 40.0, 12.0, 40)),
              "PNP", "esoubies", "native", "real",
              {"denoiser": "TV Bregman", "sigma": 40, "lambda_kz": 12, "iter": 40}),
        _spec("matirf", esoubies_config("ADMM-PnP", _admm_pnp_matirf("Gaussian", 50.0, 0.03)),
              "ADMM-PnP", "esoubies", "native", "real",
              {"denoiser": "Gaussian", "sigma": 50, "rho": 0.03}),
        _spec("matirf", esoubies_config("MCMC", _mcmc_matirf("TV Bregman", 0.01, 5.0, 300)),
              "MCMC", "esoubies", "native", "real",
              {"denoiser": "TV Bregman", "sigma": 0.01, "lambda_rr": 5, "iter": 300}),
    ]


def _mcmc():
    """MCMC — MMSE by sampling (a genuine reconstructor: it BEATS the ridge start, lambda_rr~5 not s1).
    Standard = the old weak default (TV Bregman sigma=0.02, lambda_rr=s1, iter=300). Sweeps TV Bregman /
    Gaussian / DCT per structure at the good regime (sigma=0.01, lambda_rr=5, iter=300). + deconv (excels)."""
    out = []
    for truth in TRUTHS:
        for noise in NOISES:
            out.append(_spec("matirf", matirf_config(truth, "MCMC", noise, _mcmc_matirf("TV Bregman", 0.02, MATIRF_S1, 300)),
                             "MCMC", truth, noise, "standard",
                             {"denoiser": "TV Bregman", "sigma": 0.02, "lambda_rr": MATIRF_S1, "iter": 300}))
            for den, sigma, lrr in MCMC_DENOISERS:
                out.append(_spec("matirf", matirf_config(truth, "MCMC", noise, _mcmc_matirf(den, sigma, lrr, 300)),
                                 "MCMC", truth, noise, "denoiser",
                                 {"denoiser": den, "sigma": sigma, "lambda_rr": lrr, "iter": 300}))
    for den in ("TV Bregman", "Gaussian"):                     # deconvolution: the MMSE reference
        out.append(_spec("deconv", deconv_config("MCMC", 0.02, _mcmc_deconv(den, 0.03)),
                         "MCMC", DECONV_IMAGE, 0.02, "denoiser", {"denoiser": den, "sigma": 0.03}))
    return out


def specs():
    """The curated campaign, de-duplicated by run_id."""
    seen, out = set(), []
    for spec in _adam() + _ppxa() + _admm_specs() + _pnp() + _admm_pnp() + _mcmc() + _esoubies():
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
    lines = [f"Curated campaign: {len(all_specs)} runs — truths {', '.join(TRUTHS)}"]
    for (problem, solver), n in sorted(by.items()):
        lines.append(f"  {problem:7} {solver:12} {n:4d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
