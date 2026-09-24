# Intrinsic limits — what each algorithm cannot do, and why (Étape 3)

> Status: **synthesis** of the failure, error, rejection and bad-reconstruction modes seen in
> the Phase-A atlas, read back onto the theory of `00_foundation.md` and the per-algorithm
> notes. This is the interpretive step of the benchmark plan: it turns *observed failures* into
> statements of each algorithm's **intrinsic** limit — the wall that no parameter choice moves,
> for whoever picks up this code next.
>
> Provenance is marked throughout, because it matters here:
> - **[structural]** — follows from the math + the operator's spectrum (`00_foundation.md`);
>   certain, independent of any run.
> - **[test]** — reproduced in `solvers/_tests.py` (present in the tree, re-runnable).
> - **[Phase A]** — observed in the full-resolution campaign. The raw per-run rows
>   (`benchmarks/results/`, gitignored) were lost to a worktree cleanup, so these are stated as
>   the *mode* that was seen and its *mechanism*, using the realism check's own verdict words
>   ("empty", "collapsed", "no better than ridge") — not with reconstructed numbers.
>
> Read after `00_foundation.md`. Convention: physical operator (`normalize = False`), spectrum
> `s1 = 38.9, s2 = 6.02, s3 = 0.562, s4 = 0.033`; `f` peaks ≈ 0.02 on MA-TIRF, ≈ 1 on deconv.
>
> Part of the benchmark arc: **Étapes 1 & 2** (how to use, which v1/v2 to keep) are the
> provisional pre-benchmark report in [`../benchmark/pre_benchmark.md`](../benchmark/pre_benchmark.md);
> this file is **Étape 3**; **Étape 4** (effective ranges → the atlas) is the new benchmark.

---

## 0. The three limits that are the problem's, not the algorithm's

Before the per-algorithm walls, three limits sit under **every** method on MA-TIRF. No solver
escapes them; they are the reason the reconstructions look the way they do.

1. **The null space is huge. [structural]** Per depth column H has only 2–3 determined
   singular directions (`s1, s2`, and `s3` partly: `s3² ≈ 0.3`); the other ~47 of 50 lie in the
   numerical null space (`00_foundation.md` §1.1). The data cannot see them. **Whatever those
   ~47 directions end up holding comes from the prior and from positivity — never from `g`.** So
   the ceiling on any reconstruction is set by how good the *prior* is, not by the optimizer.
   The universal failure signature is **depth collapse**: the recovered object slides in `z` or
   flattens, because depth is exactly what the data under-determine.

2. **Noise devastates MA-TIRF out of proportion. [structural, Phase A]** The determined
   directions span `s1²/s3² ≈ 5e3` in curvature. Noise entering along the weak directions
   (`s3² ≈ 0.3`, `s4² ≈ 1e-3`) is amplified by `1/s_i²` — hundreds to thousands of times. So a
   noise level that is mild on a photograph (deconv, well-conditioned) is destructive on
   MA-TIRF. In Phase A the jump from σ = 0 → 0.02 → 0.05 (Gaussian) moved several solvers from
   "usable" to the realism check's "collapsed"/"no better than ridge" — a far steeper fall than
   on deconv. **This is why the data term carries the noise level `b` by construction**
   (`00_foundation.md` §3.1): the honest response to noise is to trust the data *less*, which is
   also to lean *harder* on a prior the algorithm may not have.

3. **Photograph-scale defaults are ~40× wrong. [structural, Phase A]** `f` peaks ≈ 0.02, not 1
   (`00_foundation.md` §1.2). Any parameter "in units of `f`" — a denoiser σ on a 0–255 scale, a
   proposal noise, a sparsity threshold, an Adam learning rate, an adjoint start — is off by
   ≈ `s1 ≈ 40` if it was set for peak-1 data. This is the single most common way a method that
   works elsewhere silently fails here. Half of the v1→v2 rewrites (PnP ÷peak, MCMC σ-as-
   fraction, ridge-s2² starts) exist only to remove this one trap.

Everything below is *on top of* these three.

---

## 1. Adam / PPXA — the general MAP minimizers

**Intrinsic limit: they are only as good as the prior `R` you give them, and `R` is a weak
prior for a null space this large. [structural]**

Adam and PPXA both minimize `(1−λ)·D + λ·κ·R` exactly (PPXA to the global optimum in the limit;
Adam to it with enough budget). So they have **no failure of their own** on a well-posed
problem — and that is precisely the point: on MA-TIRF the reconstruction quality is decided
entirely by `R` and `λ`, and none of the six analytic regularizers models a real cell's depth
structure the way a learned denoiser does. Their ceiling is the ceiling of hand-built priors.

Failure modes, all at the *ends* of `λ`, all reproducible:

- **`λ → 0`: noisy / depth-collapsed.** Non-negative least squares lets noise through `s3, s4`;
  the realism check reads it as "no better than ridge" / "collapsed". [structural, Phase A]
- **`λ → 1`: empty (or flat).** The data are dropped, `f → argmin R` = 0 for L1/L2/SHV, a
  constant for TV/Tikhonov → realism check's "empty". [structural, Phase A]
- **Adam-specific: the start's scale can eat the whole budget.** From an `adjoint` start
  (~850× too large at MA-TIRF scale), Adam must walk every voxel from ~20 down to ~0.02 in
  steps of `lr` while the scheduler halves `lr` on the oscillations — the iterations run out
  first (the v1 "depth-shift" regression). **This is a limit of the finite budget, not of the
  method:** with the ridge-s2² start (default) it disappears. `lr` below `0.1·max f0` also fails
  to move within budget. [structural; the fix is the default start]

**For a successor:** Adam/PPXA are the right *engine* and the wrong *prior*. The gain on MA-TIRF
is in the regularizer (a learned/plug-and-play prior), not in a better optimizer. PPXA's only
extra limit is that it needs proximable priors; Adam takes any differentiable one.

---

## 2. ADMM (v1, soft-threshold) — sparsity as the prior

**Intrinsic limit: L1 sparsity is the wrong prior for a diffuse object, and its one useful knob
(`κ`) is unpredictable *because* of the null space. [structural, Phase A]**

The soft-threshold zeroes every voxel below `t = κ·max(g)/s1`. But the data step spreads the
object's energy across the ~47 null-space directions (`admm.md` §3.1), so the voxels of the
ridge reconstruction are all *small* — a diffuse truth can have **every** voxel below the
threshold. Then:

- **`κ` too large → empty image.** [structural, Phase A] The threshold sits above the (spread-
  out) voxel values and erases everything. This is the dominant ADMM failure on MA-TIRF, and it
  arrives *early* — the "empties-it" weight is only `τ_max = max(Hᵀg)` (`admm.md` §4.1).
- **`κ ≈ 0 → noisy / depth-collapsed`**, and then `iter` becomes an accidental regularizer
  (20 vs 2000 steps give different images) — a well-posedness smell. [structural]
- **`μ > 1.618 → oscillates / diverges (NaN).**  [test] `solvers/_tests.py`: `μ = 2.5 → NaN`;
  `μ` from 0.3→1.7 moves the solution from 21 to 6 non-zero voxels (μ is not a speed knob in
  v1 — it is coupled into the prior weight and the dual step at once).
- **The useful `κ` band cannot be predicted a priori** — it entangles `κ` with `μ`, `s1` and the
  truth's spread. This is flagged in `admm.md` §3.1 as *"the least certain prediction in the
  whole analysis"*. [structural]

**Continuous truths (`cell`) are essentially out of reach for ADMM**: L1 is the wrong model, so
there is little-to-no `κ` band. Concentrated truths (`vesicles`, `fibres`) have a band, but a
narrow one. **For a successor:** ADMM's wall is its prior (sparsity), not its solver. The
coupled `μ` is a genuine design defect — ADMMv2 (`solvers/old/`) fixed the *interpretability*
of `κ` and freed `μ` to be speed-only, but did **not** change the underlying limit that L1 is a
poor prior for cells; it also **empties the image at `κ → 1` by construction**
(`κ = 1` returns exactly zero — lasso optimality at `f = 0`, `admm.md` §4.1) [structural],
which made it fragile at the top of its own band.

---

## 3. PnP-HQS & ADMM-PnP — plug-and-play denoiser priors

These two share their prior (a denoiser) and therefore their limits; PnP uses an annealing
schedule, ADMM-PnP a dual variable.

**Intrinsic limit A: the denoiser scale problem — most denoisers are a *different prior* on
MA-TIRF than on a photograph. [structural, test]** `pnp.md` §3.1, measured relative change at
σ = 5:

| peak | Gaussian | TV Bregman | Bilateral | Wiener | DCT |
|---|---|---|---|---|---|
| 0.022 (MA-TIRF) | 0.01 | 0.20 | 0.22 | **0.84** | **0.99 (erased)** |
| 1 (deconv) | 0.01 | 0.20 | 0.09 | 0.20 | 0.40 |

So a σ tuned on one problem does not carry to the other for Bilateral / Wiener / DCT, and **DCT
is systematically rejected on MA-TIRF** (it erases the image). Only Gaussian and TV Bregman are
scale-free. The v2 ÷peak fix removes this for the *kept* PnP, but the limit is intrinsic to
naïve plug-and-play and worth stating: **a denoiser is only a transferable prior if it is fed at
the scale it was built for.**

**Intrinsic limit B: collapse under noise.** [structural, Phase A] Like every MAP method here,
PnP-HQS leans on the data step for the determined directions; when noise floods `s3, s4` and the
denoiser is not strong enough (σ too small, or `λ_kz` too small letting noise in at the end),
the reconstruction collapses. Raising σ over-smooths and erases the little the data *could*
restore (only `s1, s2, [s3]`). The usable window is genuinely narrow on noisy MA-TIRF.

**Intrinsic limit C: cost and memory on large real stacks (`esoubies`). [Phase A]** The PnP
family was the family that hit **out-of-memory / timeout on the `esoubies` real dataset** — the
3D denoiser runs on the full high-resolution volume every iteration (16 annealing steps for PnP;
one denoise per ADMM-PnP iteration), and the per-iteration denoise dominates the runtime the
time estimates repeatedly under-called. This is a scaling limit, not a correctness one: it says
the PnP family needs a memory/tiling budget before it is run on full real volumes.

**PnP vs ADMM-PnP specifically:**
- **PnP-HQS** is *not run to convergence* — it is an annealing schedule (`iter` = number of
  schedule steps, 8–30 stable; 5 is coarse). Its quality is the schedule's, so a too-short
  schedule is itself a limit. [structural]
- **ADMM-PnP** converges to a fixed point via the dual variable; its prior strength scales like
  `ρ·σ²` (`admm_pnp.md`), so `ρ` and `σ` are coupled into one effective knob — a mild version of
  ADMM's `μ` coupling.

**For a successor:** the denoiser *is* the prior and *is* the ceiling. A better MA-TIRF result
comes from a 3D, scale-robust, cell-trained denoiser — and from making the family affordable on
full real volumes (tiling / lower precision), which is currently the binding practical limit.

---

## 4. MCMC — the MMSE estimator

**Intrinsic limit: on MA-TIRF, no setting beats the ridge start — and this is structural, not a
tuning failure. [structural, Phase A]** This is the sharpest, best-established limit in the whole
benchmark, documented in full in `mcmc.md` §5.

The chain explores by perturbing `f` and accepting on the data term `D`. But the exploration
that *matters* — the ~47 undetermined depth directions — lives in D's **null space**, which by
definition does **not** change `D`. So the Metropolis test is *blind* to exactly the directions
that need deciding:

- with `β` too small, nothing is accepted → the chain freezes at its start; [structural]
- with `β` large, almost everything is accepted (90–100%) → the chain diffuses in the null space
  and the mean drifts rather than concentrates. [Phase A]

Neither regime concentrates the posterior where it matters. **`ΔD⁺ ∝ σ²/b`**, so no fixed `β`
works across σ and noise — which is why v1 MCMC "worked with very specific parameters or not at
all" (`mcmc.md` §3.3, §4). Two attempted rescues both failed, and both failures are structural:

- **Re-projecting each candidate onto data consistency** (paper-faithful) keeps `D ≈ const`,
  making the test *even more* blind → acceptance saturates. [Phase A, MCMCv3, abandoned]
- **Accepting on `D + γ·R`** did not rescue it either. [Phase A]

The paper's (Moebel & Kervrann 2020) success relies on a **small, hard-constrained** missing
region; MA-TIRF's null space is huge, so the method's core assumption does not hold here.

**Where MCMC does work: deconvolution (well-posed). [Phase A]** There the chain concentrates and
the MMSE mean beats the ridge start (TV Bregman, λ_rr = s1, σ ≈ 0.03). So MCMC is kept as the
**MMSE reference on deconv** and as an **honest negative on MA-TIRF** — it is reported as *not
beating the ridge start there*, which is itself a result.

Secondary limits: **DCT breaks the reconstruction** (rejected by the realism check, same scale
problem as §3); **`None` denoiser = no prior at all**, cannot beat its own start (a diagnostic,
never a working setting). [structural]

**For a successor:** do not spend effort making MCMC win on MA-TIRF — the null space forecloses
it. Its value is the uncertainty/MMSE view on well-posed problems (deconv) and the honest
baseline it provides on the ill-posed one.

---

## 5. Summary — the wall, the signature, the way around it

| algorithm | intrinsic wall | failure signature | can it be moved? |
|---|---|---|---|
| **Adam / PPXA** | only as good as the analytic prior `R` | `λ→0` noisy/collapsed; `λ→1` empty/flat | not by the optimizer — needs a better prior |
| **ADMM (v1)** | L1 is the wrong prior for diffuse cells; `κ` unpredictable; `μ` coupled (>1.618 → NaN) | `κ` high → **empty**; `κ`≈0 → noisy; continuous `cell` → no band | no — the wall is the sparsity prior |
| **PnP-HQS** | denoiser scale + collapse under noise; short schedule | collapse (noise) / over-smooth (σ big); OOM on `esoubies` | scale fixed by ÷peak; ceiling = the denoiser |
| **ADMM-PnP** | same denoiser ceiling; `ρσ²` coupled; cost | OOM/timeout on full real volumes | needs tiling + a better 3D denoiser |
| **MCMC** | MH test is **blind to the null space** on MA-TIRF | never beats the ridge start on MA-TIRF; freezes or saturates | **no** on MA-TIRF (structural); works on deconv |

**The one sentence.** On MA-TIRF every wall is the same wall seen from a different side — the
~47-direction null space — and the only lever against it is the **prior**: which is why the
MAP methods with a good denoiser (PnP family) are the ones with headroom, and why MCMC, whose
only test is on the data, cannot win here at all.
