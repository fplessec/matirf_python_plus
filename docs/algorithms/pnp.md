# PnP-HQS & PnPv2 — a priori analysis of the parameters

Reads on top of `00_foundation.md`. Plug-and-Play (Venkatakrishnan 2013) keeps a splitting
algorithm's structure but replaces the prior's proximal operator by **a denoiser**: the prior
is then whatever the denoiser implicitly believes an image looks like. This solver is
**Half-Quadratic Splitting (HQS)** with the DPIR annealing (Zhang 2021). It is a **MAP**-style
estimator; it does **not** use `reg` / `lambda_reg` / κ. **The denoiser analysis of §3 is
shared with ADMM-PnP and MCMC.** Physical operator (s1 = 38.9, s2 = 6.02, s3 = 0.562, s4 =
0.033). Math in plain text.

---

## 1. What PnP-HQS does — and its Bayesian reading

HQS splits `min_f ½||Hf−g||² + Φ(f)` by introducing a copy `z = f` tied with a quadratic
penalty of weight α:

```
min over f, z :  ½·||Hf − g||²  +  Φ(z)  +  (α/2)·||f − z||²
```

Alternating minimization gives: an f-step (a quadratic solve) and a z-step
`z = prox_{Φ/α}(f)`. **The Plug-and-Play idea:** replace `prox_{Φ/α}` by a Gaussian denoiser
`D_σ` at level `σ` — because the proximal operator of a prior *is* the MAP denoiser of that
prior under Gaussian noise, with `σ² ∝ 1/α`. Φ is never written; the denoiser stands in for it.

**Bayesian reading.** The f-step is the Gaussian-likelihood MAP given a prior mean z; the
z-step is "what an image should look like" imposed by the denoiser. The estimate is the fixed
point of "trust the data on the directions it determines, trust the denoiser on the rest".

Note: the data term is the plain `½||Hf−g||²`; PnP does not carry the noise level.

---

## 2. Logic, pseudo-code, and the mathematics (`solvers/pnp.py`)

```
z ← f0                                        # PnP v1 start = adjoint Hᵀg (foundation §4)
schedule α_k ↑ , σ_k ↓   over `iter` steps    (DPIR annealing, §2.2)
f ← z
repeat k = 1 … iter:
    f ← (HᵀH + α_k·I)⁻¹ ( Hᵀg + α_k·z )       # data step: fit data, pulled toward z with weight α_k
    z ← D_{σ_k}( f )                           # prior step: denoise at level σ_k
    if forced_pos:  f, z ← max(·, 0)
return f
```

### 2.1 The data step, direction by direction

```
along singular direction i :   f_i = ( s_i² / (s_i² + α) )·[LS]_i  +  ( α / (s_i² + α) )·z_i
```

- `s_i² ≫ α` → f follows the data (`[LS]_i`);
- `s_i² ≪ α` → **f keeps the denoised z_i**.

On MA-TIRF, at any reasonable α, ~47 of the 50 depth directions per column are of the second
kind: **the depth structure of the reconstruction is essentially the denoiser's** (foundation
§1.1). This is the single most important fact about PnP here.

### 2.2 The annealing schedule (`kai_zhang = True`)

```
σ_final = 1        (hard-coded in v1)
σ_k : log-spaced from `sigma` down to σ_final
α_k = λ_kz / σ_k²   (so α_k·σ_k² = λ_kz constant),  rising to λ_kz
```

It starts weak-data / strong-denoise and ends data-trusted at weight `λ_kz` with a light
denoising at level 1. The schedule is what makes plain HQS (no dual variable) usable. **Caveat
in v1:** `σ_final = 1` is hard-coded, i.e. PnP always assumes a final noise of `1/255 ≈ 0.4%`
of the peak, whatever the measurement — wrong when the real noise differs (PnPv2 fixes it).

---

## 3. The denoisers (shared with ADMM-PnP and MCMC)

The project contract (`solvers/denoising.py`): a denoiser is applied as `D(255·f, σ)/255`,
**assuming f ∈ [0,1]**. σ is a level on the 0–255 scale.

| denoiser | acts through σ as | 3D? | depends on the image's intensity scale? |
|---|---|---|---|
| Gaussian | spatial width `clip(σ/25, 0.3, 2.5)` px | yes (uses δ) | no |
| TV Bregman | data weight `clip(3/σ, 0.005, 1)` | **no** (slice by slice) | no |
| Bilateral | intensity tolerance `2σ` (+ bounded width) | yes | **strong** |
| Wiener | gain `max(var − σ², 0)/var` | yes | **very strong** |
| DCT | threshold `σ·√(2 log N)` | yes | **very strong** (broken → excluded) |

Two MA-TIRF-specific consequences (foundation §1.1–1.2):

- **TV Bregman works slice by slice** — it never smooths along z, yet z is exactly what the
  denoiser must structure (§2.1). Expected: poor depth structure on MA-TIRF (it can still be a
  fine 2D denoiser for deconvolution).
- **Gaussian is a no-op for σ ≤ 7.5** (width pinned at 0.3 px): the default σ = 5 with a
  schedule going down to 1 means the Gaussian denoiser does *nothing*.

### 3.1 The scale problem — why σ does not transfer between problems

The contract assumes `f ∈ [0,1]`. True for deconvolution (a photograph, peak ≈ 1), **false for
MA-TIRF, where f peaks ≈ 0.02** (foundation §1.2): the image reaches the denoiser on a 0–5
scale while σ is read on 0–255, so a given σ is effectively ~40× too strong.

*Measured* — relative change `||D(f)−f|| / ||f||` at σ = 5, same noisy image, peak 0.022
(MA-TIRF) vs peak 1 (deconvolution):

| | Gaussian | TV Bregman | Bilateral | Wiener | DCT |
|---|---|---|---|---|---|
| peak 0.022 | 0.01 | 0.20 | 0.22 | **0.84** | **0.99** (erased) |
| peak 1 | 0.01 | 0.20 | 0.09 | 0.20 | 0.40 |

So for **Bilateral, Wiener, DCT the same σ is a different prior on the two problems** —
parameters tuned on one do not carry to the other, and it is why DCT was systematically
rejected on MA-TIRF. **Gaussian and TV Bregman are scale-free.** (PnPv2's ÷peak fix removes
this, §5.)

---

## 4. The parameters of PnP v1

### 4.1 `denoiser` — the prior *(critical: it shapes the ~47 null-space directions)*

The single most consequential choice (§2.1). On MA-TIRF the denoiser must structure the depth,
so it must be 3D and scale-robust: **Gaussian** (with σ large enough, §4.2) or **Bilateral**
(3D, but scale-dependent — §3.1). TV Bregman is 2D-only (poor in z). Not a Phase-A "sweep":
a discrete axis, compared per truth.

### 4.2 `sigma` (σ) — the initial denoising level *(critical)*

The schedule goes from `sigma` down to σ_final (= 1 in v1).

- `σ → 0` : no denoising → no prior → the null space keeps the start z0 (§2.1) → the start's
  depth structure plus least squares.
- `σ` large : early iterates strongly smoothed; the schedule relaxes it, but too large erases
  structures the data cannot restore (only s1, s2, [s3] are recoverable).
- **for Gaussian, σ ≤ 7.5 does nothing** (§3).

**Predicted useful range: Gaussian σ ∈ [10, 50], TV Bregman σ ∈ [2, 15]; Bilateral / Wiener no
fixed range that carries between problems** (§3.1). **Phase A:** σ sweep per denoiser (and,
for the scale-dependent ones, on both problems to expose the shift).

### 4.3 `lambda_kz` — the final data weight *(critical)*

`λ_kz` is the last iteration's α; by §2.1 it fixes which directions follow the data at the end
(`s_i² ≫ λ_kz`), and it sets the constant `α_k·σ_k² = λ_kz` along the whole schedule.

- `λ_kz → 0` : only the data at the end → noise enters along s3, s4 (`s3²≈0.3, s4²≈1e-3`).
- `λ_kz ≫ s2² = 36` : even the determined directions follow the denoiser.

**Predicted range on MA-TIRF: λ_kz ∈ [s4², s3²] ≈ [1e-3, 0.3]** (trust s1, s2, and s3 partly).
The default 0.23 ≈ s3² is at the top of it. **Phase A:** `λ_kz ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
3e-1}`.

### 4.4 `iter` — number of annealing steps *(comfort, within limits)*

The schedule is spread over `iter` steps: more steps = a finer annealing, **not** a longer run
to a fixed objective (HQS-with-schedule is not run to convergence). DPIR uses 8–40.
**Predicted: stable for iter ∈ [8, 30]; 5 (default) is coarse.** Phase A: a small sweep {8, 16,
30}, not a per-truth axis.

### 4.5 Fixed: `kai_zhang` (on — the schedule is what makes HQS usable), `forced_pos` (on),
`delta` (estimated). Not offered: `init` (v1 starts from adjoint, ~850× too large on MA-TIRF —
§5). None are Phase-A axes.

---

## 5. PnPv2 — scale-free denoising, noise-aware schedule, ridge start (`solvers/pnp_v2.py`)

Same chain, the fixes that make the parameters transfer:

| | PnP | PnPv2 |
|---|---|---|
| denoiser input | `255·f` — assumes f ∈ [0,1] | `255·f/peak`, peak = the start's peak → **scale-free** (§3.1 removed) |
| σ_final | 1 (hard-coded) | the measurement's noise level relative to its peak, estimated from g |
| δ (Gaussian/Bilateral) | 1 by default | the operator's estimate when empty |
| start | adjoint Hᵀg (~850× too large) | **ridge, λ_rr = s2² by default** (right scale, foundation §4) |
| default σ, iter | 5, 5 | 25, 16 |

**Why the ÷peak matters:** it puts every image on the 0–255 scale the denoiser expects, so a σ
tuned once works at any data scale (measured: multiplying H's gain by 50 leaves PnPv2's
reconstruction exactly divided by 50; PnP's changes by 49%). **Why ridge s2² start:** §2.1 says
the null space keeps z0, so a wrong-scaled adjoint start pollutes every undetermined direction
from the first step; a ridge s2² start is at the right scale and already contains the
determined components.

**Open point (unchanged):** `λ_kz` is still compared to the eigenvalues `s_i²` (§2.1), so it
does not transfer across operators of different gain. Expressing it relative to s1² would make
it gain-free, but the best value would still differ between a discrete spectrum (MA-TIRF) and a
continuous one (deconvolution).

**Phase A (PnPv2, the version to trust):** `denoiser ∈ {Gaussian, Bilateral}` (3D) × `σ` swept
in its range × `λ_kz ∈ [1e-3, 0.3]`; `iter` fixed ≈ 16; start = ridge s2². Verify σ transfers
between problems for Gaussian, and that PnPv2 ≥ PnP on MA-TIRF.

---

## 6. Summary — the Phase-A axes

| param | role | moves solution or speed? | predicted range | Phase-A values |
|---|---|---|---|---|
| `denoiser` | the implicit prior (shapes the null space) | **solution** | Gaussian / Bilateral (3D); TV Bregman ignores z | discrete, per truth |
| `sigma` | initial denoising level | **solution** | Gaussian [10,50], TV Bregman [2,15]; others not transferable | σ sweep per denoiser |
| `lambda_kz` | final data weight (which directions follow data) | **solution** | [1e-3, 0.3] | {1e-3 … 3e-1} |
| `iter` | annealing steps | speed (coarse↔fine) | [8, 30] | {8, 16, 30} |
| `kai_zhang`, `forced_pos`, `delta`, `init` | schedule / positivity / anisotropy / start | fixed | on / on / estimated / ridge s2² | — |

**Take-away:** PnP's reconstruction *is* its denoiser applied to the null space, so `denoiser`
+ `σ` (its strength) + `λ_kz` (how much data at the end) are the real axes. On MA-TIRF, use a
**3D** denoiser (Gaussian / Bilateral), **σ large enough to act** (Gaussian > 7.5), and PnPv2's
**scale-free ÷peak + ridge start** — otherwise the default (adjoint start, σ = 5, TV Bregman
slice-by-slice) is mis-scaled on every count.
