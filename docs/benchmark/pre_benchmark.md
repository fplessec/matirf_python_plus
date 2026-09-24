# The pre-benchmark — Étapes 1 & 2 (provisional)

> **Status: PRE-BENCHMARK, provisional.** These are the findings of the first full-resolution
> campaign (≈940 runs, run 2026-09) over the **complete v1 + v2 solver set** — the run that
> guided the solver de-duplication and the a priori notes. Its raw per-run output
> (`benchmarks/results/`, gitignored) was **lost to a worktree cleanup** (see the memory note
> `campaign-data-lost`), so the numbers below are **transcribed from the analysis, not
> re-derivable**. Treat them as a *pre-benchmark*: strong enough to have driven real decisions
> (the dedup, the parameter atlas), but **to be re-confirmed by the Étape 4 benchmark** on the
> current 6-solver set. Where a claim is structural (certain), it lives in
> [`../algorithms/limits.md`](../algorithms/limits.md) (Étape 3), not here.
>
> The four-step arc: **Étape 1** how to *use* each algorithm · **Étape 2** which of each v1/v2
> pair to *keep* · **Étape 3** each algorithm's *intrinsic limits* (→ `limits.md`) · **Étape 4**
> theoretical vs *effective* parameter ranges → the atlas (the new benchmark).

The pre-benchmark ran the full v1+v2 set: ADAM, PPXA, ADMM, **ADMMv2**, PnP, **PnPv2**,
ADMM-PnP, **ADMM-PnPv2**, MCMC, **MCMCv2**. On MA-TIRF (two synthetic truths
`cell_fibres_vesicles`, `vesicles` at noise 0 / 0.02 / 0.05, plus the real `esoubies`) and one
deconvolution image. Since then the losers were retired to `solvers/old/` (commit `09b89cc`),
so the surviving six are **ADAM, PPXA, ADMM, PnP (=former v2), ADMM-PnP, MCMC (=former v2)**.

---

## Étape 1 — how to use each algorithm (rejects, noise, image type, performance)

### A) Rejections — who, with which parameters, and why

Rejection rate per solver (the realism check refusing the reconstruction):

| solver | % reject | dominant cause |
|---|---|---|
| ADMMv2 | 97 % | *"f is zero everywhere"* — empties the image at every κ tested |
| MCMC v1 | 70 % | *"depth shifted / f ≈ start / identical planes"* — stuck |
| ADAM | 28 % | over-regularization (below) |
| ADMM-PnP / v2 | 10–11 % | fails mostly at high noise |
| ADMM / PPXA | 6–11 % | rare |
| PnP, PnPv2, MCMCv2 | 0 % | always produce a "realistic" reconstruction |

Rejection vs parameter — the key to *good use*:

- **ADAM:** rejection climbs with λ — 0/6 at λ=0, then 4/12 (λ=.02) → 7/12 (λ≥.2).
  Over-regularizing empties / over-smooths. → **use λ ≤ 0.1.**
- **ADMMv2:** 6/6 rejected at every κ ∈ [0.02, 0.5] → the threshold `τ = κ·max(Hᵀg)` is ~20×
  too strong on MA-TIRF (diffuse recon). → **κ ≪ 0.02 required** (structural, see `limits.md`).
- **MCMC v1:** 12/12 rejected at every σ → rejected whatever the setting on MA-TIRF.
  → **do not use it there.**
- **ADMM:** slight over-rejection at low κ (under-thresholding → noise); κ=0.3 never rejected.

### B) Robustness to noise (best NMSE, σ = 0 → 0.05)

| solver | cell_f_v | vesicles | robustness |
|---|---|---|---|
| ADMM-PnP | 0.05 → 0.37 | 0.05 → 0.96 | most robust on continuous structure |
| ADAM / TV | 0.45 → 0.67 | 0.06 → 0.70 | good (degrades little in absolute) |
| MCMCv2 | 0.42 → 0.78 | 0.35 → 0.84 | medium |
| ADMM | 0.04 → 0.91 | 0.02 → 0.94 | collapses (sparsity + noise) |
| PnP / PnPv2 | 0.24 → 0.95 | 0.15 → 0.97 | collapses |

Which parameters help under noise: **raise the regularization** — the best setting grows with
the noise (ADAM λ 0→0.05, ADMM κ 0.01→0.3, ADMM-PnP ρ→0.1). Noise demands a stronger prior
(consistent with the theory: the data term already trusts a noisy measurement less).

### C) Image type — the prior must match the structure

| structure | no noise | with noise |
|---|---|---|
| `vesicles` (sparse, points) | ADMM 0.02 (sparsity, near-perfect) | ADAM/TV (0.30, 0.70) |
| `cell_fibres_vesicles` (continuous / composite) | ADMM 0.04 | ADMM-PnP (denoiser: 0.17, 0.37) |

→ **sparse object → sparsity (ADMM) or TV; continuous object → a denoiser (ADMM-PnP).**
Three metrics are exploitable for the report: `nmse` (global error), `depth_error_nm` (axial
localization — the MA-TIRF question: 5 nm at 0 noise for ADMM, up to 60–100 nm under noise),
and `stack_recovery` (axial super-resolution).

### D) Performance — which parameters, and why

- **ADAM:** median-best NMSE at λ ≈ 0.1–0.2 (0.64); worst at λ=0 (0.75) and λ=0.5 (0.88).
  Too little → noise/kernel; too much → over-smoothing.
- **ADMM-PnP / v2:** strongly monotone in ρ — ρ=0.001 → 0.92, ρ=0.1 → 0.36. ρ drives data
  anchoring and prior strength (∝ ρσ²): stronger = better on the ill-posed problem.
  → **push ρ ≥ 0.1** (possibly beyond — to test in Étape 4).
- **MCMCv2:** σ=0.02 (0.54) better than σ=0.05 (0.68) → **low σ** (less over-smoothing, f≈0.02).
- **ADMM:** κ=0.3 best median (0.73) — a stronger threshold on average (dominated by the
  noisy cases).
- **PnP-HQS:** λ_kz nearly without effect (~0.98) → **poor on MA-TIRF whatever the setting.**

### Usage guide (actionable output of Étape 1)

- **ADAM** — TV/SHV, λ≈0.1 (never >0.3); λ=0 only on clean data. The most versatile.
- **ADMM** — κ 0.01–0.3 (raise with noise), μ 0.5–1; unbeatable at 0 noise on sparse objects.
- **ADMM-PnP** — Gaussian/Bilateral σ 25–50, ρ ≥ 0.1; best under noise on continuous structure.
- **MCMC (former v2)** — σ=0.02, TV Bregman / Bilateral — the usable MMSE.
- **Avoid** — ADMMv2 (κ empties the image), MCMC v1 (rejected on MA-TIRF), PnP-HQS under noise.

---

## Étape 2 — v1 vs v2: which to keep, which to delete

**Choice rule:** for each pair, keep the non-dominated version. A version is deletable if the
other (a) equals or beats it in NMSE across structures × noises, (b) is as robust
(rejects/failures), and (c) is as transferable (scale-free). If v2 dominates → delete v1 (its
reference value stays in tag `v1.0`). If v2 adds nothing → delete v2. A sound-but-mistuned v2
gets re-tuned, not deleted.

### Head-to-head (best NMSE per cell; esoubies failures; deconv)

| pair | cell_f_v (0/.02/.05) | vesicles (0/.02/.05) | esoubies fail | deconv | verdict |
|---|---|---|---|---|---|
| ADMM | 0.04 / 0.61 / 0.91 | 0.02 / 0.69 / 0.94 | 0 | — | works |
| ADMMv2 | rejected | rejected | 0 | — | unusable as-is (κ empties the image) |
| PnP | 0.24 / 0.89 / 0.95 | 0.15 / 0.96 / 0.97 | 3 | — | ok, OOM on esoubies |
| PnPv2 | 0.29 / 0.76 / 0.89 | 0.12 / 0.91 / 0.97 | 0 | — | ≥ PnP under noise + no OOM + scale-free |
| ADMM-PnP | 0.05 / 0.17 / 0.37 | 0.05 / 0.59 / 0.96 | 3 | — | slightly better NMSE |
| ADMM-PnPv2 | 0.08 / 0.24 / 0.39 | 0.05 / 0.55 / 0.93 | 5 | — | scale-free, a bit worse + more OOM |
| MCMC | rejected | rejected | 0 | 0.156 | rejected on MA-TIRF |
| MCMCv2 | 0.42 / 0.51 / 0.78 | 0.35 / 0.55 / 0.84 | 0 | 0.029 | works everywhere |

### Verdicts per pair — and what was executed

- **MCMC → keep v2, delete v1.** Clear-cut: v1 is rejected across all MA-TIRF (stuck); v2 works
  everywhere and beats it 5× on deconv (0.029 vs 0.156). v2 dominates. **[done: `09b89cc`]**
- **PnP → keep v2, delete v1.** v2 is ≥ in NMSE under noise (0.76 vs 0.89; 0.89 vs 0.95), no OOM
  on esoubies (0 vs 3 failures), scale-free (transfers to deconv). Dominates on robustness,
  equals/beats on quality. **[done: `09b89cc`]**
- **ADMM-PnP ↔ ADMM-PnPv2 → tight, neither dominated.** v1 marginally better in NMSE (0.17 vs
  0.24 on the continuous truth) and fails less on esoubies (3 vs 5); v2 brings scale-free.
  **[decided: kept v1 (`09b89cc`) — the small NMSE edge over homogeneity.]**
- **ADMM ↔ ADMMv2 → the delicate one.** ADMMv2 is unusable as-is (rejected everywhere; κ∈[0.02,
  0.5] empties the image) — but that is a *bad default band*, not a design flaw (its scale-free
  κ is cleaner than v1's entangled threshold; it needs κ≪0.02 on MA-TIRF). Options were
  (a) re-tune it, or (b) keep the proven v1. **[decided: kept v1 (`09b89cc`); ADMMv2 retired to
  `solvers/old/`.]**

**Outcome (executed in `09b89cc`):** one solver per algorithm. Retired to `solvers/old/`:
MCMC v1, PnP v1, ADMM-PnPv2, ADMMv2. The `v1.0` tag preserves every v1 reference.

---

## Étape 3 — intrinsic limits

Turned into structural statements (largely certain, theory-backed) in
[`../algorithms/limits.md`](../algorithms/limits.md). One line: on MA-TIRF every wall is the same
wall — the ~47-direction null space — and the only lever is the prior.

## Étape 4 — next

Theoretical vs **effective** parameter ranges → the atlas, from a **new** benchmark on the
current 6-solver set (`benchmarks/campaign.py`), designed to confirm/refute the pre-benchmark's
provisional numbers above and the per-note predicted ranges. Design TBD.
