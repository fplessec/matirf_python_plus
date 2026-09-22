# MCMC & MCMCv2 — a priori analysis of the parameters

Reads on top of `00_foundation.md`; denoisers as in `pnp.md` §3. MCMC is the **only MMSE**
estimator (a posterior *mean*, not a mode) and the only sampler; it is the only non-Adam/PPXA
method also run on deconvolution. Its design follows **Moebel & Kervrann (2020)**, "A Monte
Carlo framework for missing-wedge restoration". Physical operator (s1 = 38.9, s2 = 6.02,
s3 = 0.562, s4 = 0.033); f peaks ≈ 0.02 on MA-TIRF. Math in plain text.

> **This note corrects the earlier one on the central point.** The goal is **not** a high
> acceptance rate. Per the paper, the chain should be **selective** (accept ~0.25–0.6 of
> proposals), so that it *keeps the few proposals that approach the data and rejects the rest*,
> drifts toward plausible reconstructions, and their average is the MMSE. A ~100% acceptance is
> the **degenerate regime** (the paper warns: for a large temperature "all proposals are
> accepted and we fall back to denoise-and-average"). §4 records what the earlier MCMCv2
> calibration actually did, and why it missed this.

---

## 1. What MCMC does — and its Bayesian reading

Every other solver returns the **MAP** (the mode, `argmin` of an energy). MCMC returns the
**MMSE** — the posterior mean:

```
f_MMSE = E[f | g] = ∫ f · p(f|g) df
```

the minimizer of the expected squared error. The integral is intractable (thousands of
dimensions), so it is approximated by Markov-Chain Monte Carlo: draw correlated samples from
`π(f) ∝ p(f|g)` with Metropolis-Hastings, discard a burn-in, average the rest. A MAP estimate
can sit on a sharp, unrepresentative peak; the MMSE mean is smoother, and the **acceptance
rate** reports whether the chain explored (selective) or stalled/saturated.

The prior enters through the **denoiser** (there is no explicit R): the proposal denoises, so
the samples it keeps are the ones the denoiser considers "image". Without a denoiser the chain
has no prior and cannot beat its start.

### 1.1 The paper's mechanism, and ours

Moebel & Kervrann (missing wedge): `z = D( P_W(f + noise) )` — perturb, **project** onto data
consistency (keep the observed Fourier coefficients exactly, fill the unobserved ones), then
denoise. Every candidate stays data-consistent; the noise only lives in the *unobserved*
region, which the denoiser shapes; the MH test selects among these.

Our generalization (any operator H) replaces the hard Fourier projection by a **ridge pull**
toward the data. This is the one structural difference, and it matters (§5).

---

## 2. Logic, pseudo-code, and the mathematics (`solvers/mcmc.py`, `mcmc_v2.py`)

```
f ← f0
repeat k = 1 … T:
    z ← f + (HᵀH + λ_rr·I)⁻¹ · Hᵀ·(g − H·f)      # pull toward data consistency (ridge step)
    z ← z + σ·ε ,  ε ~ N(0, I)                    # explore
    z ← D_σ( max(z, 0) )                           # positivity, then the denoiser (the prior)
    ΔD ← D(z) − D(f)
    accept z with probability  min(1, exp(−ΔD / β))   # Metropolis on the data fidelity
estimate = mean of the states after a burn-in       (v2; v1 averages accepted samples)
```

### 2.1 The acceptance is a SELECTOR, and β sets how strict

A better candidate (`ΔD < 0`) is always accepted; a worse one (`ΔD > 0`) with probability
`exp(−ΔD/β)`. So β decides how often a *worsening* move is kept, i.e. how much the chain is
allowed to explore away from the data:

- `β ≪ typical ΔD⁺` : essentially no worsening move accepted → the chain freezes at its start
  (acceptance → 0);
- `β ≫ typical ΔD⁺` : every move accepted, worsening or not → **denoise-and-average, no
  selection** (acceptance → 1). The degenerate regime.
- **selective regime (the target): β of the order of the typical `ΔD⁺`, acceptance ≈ 0.25–0.6**
  (the paper's band; theory: 0.234 for a Gaussian random walk, up to ~0.574).

Because `D = ||Hf−g||²/(2·b·n_g)` (foundation §3.1), a perturbation of size σ moves Hf by ~s·σ,
so

```
ΔD⁺ ~ s²·σ² / (b·n_g)   ∝ σ²   and   ∝ 1/b
```

A fixed β tuned for one σ / one noise level is off by orders of magnitude at another. The
practical β is therefore **whatever puts the acceptance in 0.25–0.6** at the current σ, b — so
β should be **calibrated to a target acceptance rate**, not fixed and not set to saturate.

### 2.2 The data step and λ_rr

`(HᵀH + λ_rr·I)⁻¹` corrects the directions with `s_i² ≫ λ_rr` toward the data and leaves the
rest (foundation §4). On MA-TIRF only 2–3 depth directions are determined, so λ_rr decides how
many the chain re-anchors each step:

- `λ_rr ∈ [s3², s2²] ≈ [0.3, 36]` keeps the determined directions, cuts the noise-dominated
  ones;
- `λ_rr ≪ s4²` amplifies noise along the poorly determined directions;
- `λ_rr ≫ s1²` (v1's default 1e4) barely pulls → the chain leans on the denoiser alone.

MCMCv2's automatic λ_rr = s1 = 38.9 (≈ s2², top of the window: keeps s1 fully, s2 half).

---

## 3. The parameters

### 3.1 `denoiser` — the prior *(critical)*

Same catalogue and MA-TIRF caveats as `pnp.md` §3. TV Bregman is the most robust across noise
here (measured); Wiener is good at low noise and degrades; `None` = no prior (a diagnostic,
never a working setting). It must be 3D-capable to structure the depth — though on MA-TIRF the
gain from any denoiser is limited (§5).

### 3.2 `sigma` (σ) — exploration **and** denoising strength *(critical)*

One number does two jobs: the std of the perturbation *and* the denoiser's level. In **v1** it
is absolute (default 0.05) — but f peaks ≈ 0.02 on MA-TIRF (foundation §1.2), so the default
perturbation is **~2.5× the whole signal** (and it means something different on a photograph,
peak 1). MCMCv2 makes it a **fraction of the peak** (default 0.03).

- `σ → 0` : no exploration, no denoising → the chain stays at its start.
- `σ` large : over-smoothing, and `ΔD⁺ ∝ σ²` explodes → acceptance collapses (§2.1).
- **Predicted useful range: σ ≈ 0.02–0.05 of the peak**, its optimum growing with the noise.

### 3.3 `beta` (β) — the temperature / selectivity *(critical)*

The switch of §2.1. **The correct target is a selective acceptance (0.25–0.6), not a value
fixed in advance:** since `ΔD⁺ ∝ σ²/b`, the right β depends on σ and the noise, so it should be
**calibrated to hit that acceptance band** on a short warm-up. The exact Bayesian posterior
would be β = 1/n_g (a per-pixel log-likelihood), far too small (frozen); the *practical* β is
larger, set by the acceptance target.

### 3.4 `lambda_rr` — data-step ridge *(mostly fixed by a rule)*

§2.2. **Useful window `[s3², s2²] ≈ [0.3, 36]`**; MCMCv2 defaults to the automatic s1 = 38.9
(≈ s2²). v1's 1e4 barely pulls.

### 3.5 `max_iter` (T), `K`, `init`

The chain needs a burn-in before its states represent the posterior; then the post-burn-in
mean beats both the last state (one noisy sample) and the burn-in-included mean (biased toward
the start). **Predicted: a plateau by ~150 iterations**; use T ≈ 300 with a 25% burn-in. `K` is
only the logging cadence. `init`: not critical (the burn-in erases the start); ridge s2² by
default.

---

## 4. MCMCv2 — the proposals, and the calibration that MISSED the selective regime

`solvers/mcmc_v2.py`. Its intended fixes are sound; one is not:

| | MCMC | MCMCv2 |
|---|---|---|
| σ | absolute (default 0.05) | **fraction of the peak** (0.03) — scale-free ✓ |
| λ_rr | 1e4 (barely pulls) | automatic s1 ✓ |
| estimate | mean of accepted samples, no burn-in | mean of all states after 25% burn-in ✓ |
| β | raw number, hand-matched | `β = τ · median|ΔD|` over the first proposals ✗ |

**Why the β calibration is wrong (measured).** Tying β to `median|ΔD|` was meant to remove the
σ²/b coupling, and it does — but it sets β *at the scale of a typical move*, which on our chain
lands acceptance at **99–100%** (measured: target-agnostic, τ from 0.3 to 10 all give ~100%).
That is the degenerate denoise-and-average regime the paper warns against, **not** the selective
0.25–0.6 the method needs. The philosophy-correct calibration is to **tune β to a target
acceptance rate** (e.g. 0.4), on a short warm-up — which MCMCv3 was built to test and then
dropped (§5).

---

## 5. The honest limit: MCMC on MA-TIRF (from the investigation, 2026-09)

Measured, sweeping β, σ, λ_rr and the denoiser, on cropped MA-TIRF and deconvolution patches:

- **On MA-TIRF, no setting beats the ridge start** (best NMSE ≈ the start's; usually worse).
  The reason is structural: with only 2–3 determined directions, the exploration lives in D's
  **null space**, which by definition does **not** change D — so the MH test (on D) is *blind*
  to it, the acceptance either freezes or saturates regardless of β, and the mean drifts rather
  than concentrates. (Re-projecting each candidate onto data consistency, à la the paper, keeps
  D ≈ const and makes the test *even more* blind — acceptance saturates — so it does not help
  here either. Accepting on `D + γ·R` did not rescue it.) MCMC's success in the paper relies on
  a **small, hard-constrained** missing region; MA-TIRF's null space is huge.
- **On deconvolution (well-posed), MCMC works**: the chain concentrates and the MMSE mean beats
  the ridge start (e.g. TV Bregman, λ_rr = s1, σ ≈ 0.03).

**Consequence for the benchmark:** MCMC is worth running as the **MMSE reference on
deconvolution** and as an honest negative on MA-TIRF (report that it does not beat the ridge
start there). MCMCv3 (a paper-faithful reproject + target-acceptance β) was built, measured to
give no MA-TIRF gain and to still saturate, and **abandoned** — kept only in `docs/old` history.

---

## 6. Summary — the Phase-A axes

| param | role | moves solution or speed? | predicted range | Phase-A values |
|---|---|---|---|---|
| `denoiser` | the prior (there is no R) | **solution** | TV Bregman robust; Wiener low-noise; None = none | discrete |
| `sigma` | exploration × denoising | **solution** | 0.02–0.05 of the peak | {.01, .02, .03, .05} × peak |
| `beta` | temperature = selectivity | **solution** (via the acceptance) | the β giving acceptance 0.25–0.6 | tuned to a target rate, per (σ, noise) |
| `lambda_rr` | data-step ridge | by rule | [s3², s2²] ≈ [0.3, 36] | auto = s1 (≈ s2²) |
| `max_iter` (T) | budget + burn-in | speed | ≳ 150 (use 300, 25% burn-in) | fixed |
| `init` | start | no | ridge s2² | — |

**Take-away:** MCMC is about **which denoiser** and **how much to explore (σ)**, with **β tuned
to a *selective* acceptance (0.25–0.6)** — not to saturate. It is a genuine MMSE method on
well-posed problems (deconvolution); on severely ill-posed MA-TIRF the sampler cannot beat the
ridge start, which the atlas should record rather than hide.
