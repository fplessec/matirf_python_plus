# ADMM-PnP & ADMM-PnPv2 — a priori analysis of the parameters

Reads on top of `00_foundation.md`; the **denoisers, their scale problem and their MA-TIRF
limits are in `pnp.md` §3** (shared, not repeated). ADMM-PnP is the augmented-Lagrangian
counterpart of PnP: the same "data vs denoiser" split, but with a **dual variable** instead of
an annealing schedule. **MAP**-style; does not use `reg` / `lambda_reg` / κ. Physical operator
(s1 = 38.9, s2 = 6.02, s3 = 0.562, s4 = 0.033). Math in plain text.

---

## 1. What ADMM-PnP does — and its Bayesian reading

ADMM splits `min ½||Hx − g||² + Φ(v)` with `x = v`, and replaces the prox of the (unknown)
prior Φ by a denoiser. Same MAP intent as PnP (`pnp.md` §1): the data decide the directions
they determine, the denoiser decides the rest — but here a dual variable u accumulates the
mismatch, so ρ and σ can stay **fixed** (no coarse-to-fine schedule).

---

## 2. Logic, pseudo-code, and the mathematics (`solvers/pnp_admm.py`)

```
x ← f0                                   # v1 start = adjoint Hᵀg (foundation §4)
v ← x ;  u ← 0
repeat k = 1 … iter:
    x ← (HᵀH + ρ·I)⁻¹ ( Hᵀg + ρ·(v − u) )   # data step: fit data, pulled toward v − u
    v ← D_σ( x + u )  ,  then max(v, 0)      # prior step: denoise at a FIXED level σ
    u ← u + (x − v)                          # dual step (standard: no ρ factor)
return v
```

### 2.1 The data step, and why ρ frees the dual

Direction by direction (as `pnp.md` §2.1 with α → ρ): `s_i² ≫ ρ` follow the data, `s_i² ≪ ρ`
keep `v − u`. On MA-TIRF the depth structure is again essentially the denoiser's.

The dual step is the **standard** `u ← u + (x − v)` (contrast ADMM v1's `η ← η + μ(u−f)`, which
capped μ < 1.618). So ADMM-PnP has **no stability cap on ρ**: any ρ > 0 is admissible. (Formal
convergence needs a non-expansive denoiser or an increasing ρ — Chan 2017; in practice the
iterates settle or oscillate slightly, read from the logged `||x−v||/||v||`.)

### 2.2 The prior strength is the product ρ·σ²

If the denoiser were the exact MAP denoiser of a prior `λ·Φ` under Gaussian noise of level σ,
its step would be `prox_{λΦ·σ²}`; ADMM needs `prox_{Φ/ρ}`. Matching them gives

```
λ_implicit  ∝  ρ · σ²          (the prior's overall strength)
```

- raising **ρ** at fixed σ: strengthens the prior **and** trusts fewer data directions (§2.1);
- raising **σ** at fixed ρ: strengthens the prior **only**.

Unlike textbook (convex) ADMM where ρ changes only the speed, **here ρ changes the solution**:
PnP-ADMM has no fixed objective, its fixed point depends on (ρ, σ).

---

## 3. The parameters of ADMM-PnP v1

### 3.1 `denoiser` — the prior *(critical)*

Same catalogue and same MA-TIRF caveats as PnP (`pnp.md` §3): TV Bregman is slice-by-slice
(poor in z); Gaussian is a no-op for σ ≤ 7.5; Bilateral / Wiener / DCT depend on the image's
intensity scale (`pnp.md` §3.1), which at MA-TIRF's peak ≈ 0.02 makes a given σ far stronger
than on a photograph. Use a 3D, scale-robust denoiser (Gaussian, or Bilateral via v2's ÷peak).

### 3.2 `sigma` (σ) — the denoising level *(critical)*

Fixed for every iteration (default 15, on the 0–255 scale). By §2.2 it is the prior's strength
at fixed ρ.

- `σ → 0` : no denoising → no prior → the undetermined directions keep the start (badly scaled
  in v1, §5).
- `σ` large : over-smoothing; only s1, s2, [s3] are restorable by the data.

**Predicted useful range:** Gaussian σ ∈ [10, 50] (default 15 ≈ a 0.6 px kernel), TV Bregman
σ ∈ [2, 15]; Bilateral / Wiener at MA-TIRF's scale far below their deconvolution values (not
transferable — `pnp.md` §3.1). **Phase A:** σ sweep per denoiser.

### 3.3 `rho` (ρ) — the penalty *(critical: three effects at once)*

By §2.1–2.2: the data-step ridge (which directions follow the data), the prior strength `ρ·σ²`,
and the speed.

- `ρ → 0` : the data dominate; noise enters along s3, s4; the prior barely acts.
- `ρ ≫ s2² = 36` : even the determined directions follow the denoiser; the data are ignored.

**Predicted range on MA-TIRF: ρ ∈ [s4², s3²] ≈ [1e-3, 0.3]** (trust s1, s2, and s3 partly) — the
same logic and the same range as PnP's `λ_kz`. The default 1.0 is just above it (already hands
s3 to the denoiser). **Phase A:** `ρ ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}` at a fixed σ, and
a paired sweep along `ρ·σ² = const` to check §2.2.

### 3.4 `iter` — iterations to the fixed point *(comfort, within limits)*

With fixed (ρ, σ) the iterates approach a fixed point or oscillate around it (§2.1).
**Predicted: a plateau within 20–50 iterations.** The default 20 is at the low end; **Phase A
should give ≈ 50** and read the plateau from the logged `||x−v||/||v||`.

### 3.5 Fixed: `forced_pos` (on), `delta` (estimated). Not offered: `init` (v1 = adjoint,
~850× too large; it enters twice, as x and as the v of the first data step — §5).

---

## 4. ADMM-PnPv2 — scale-free, noise-aware σ, ridge start (`solvers/pnp_admm_v2.py`)

| | ADMM-PnP | ADMM-PnPv2 |
|---|---|---|
| denoiser input | `255·(x+u)` — assumes f ∈ [0,1] | `255·(x+u)/peak` → **scale-free** (`pnp.md` §3.1 removed) |
| σ | fixed, default 15 | empty ⇒ **3 × the measurement's noise level** (relative to its peak) |
| δ | 1 by default | the operator's estimate when empty |
| start | adjoint Hᵀg (~850× too large) | **ridge, λ_rr = s2² by default** (right scale) |
| default ρ, iter | 1.0, 20 | 0.1, 50 (inside the predicted ranges) |

The factor 3 tying σ to the noise is a **first guess** (Phase A settles it). Open point
(unchanged): ρ is still compared to the eigenvalues `s_i²`, so it does not transfer across
operators of different gain. **Phase A (ADMM-PnPv2, the version to trust):**
`denoiser ∈ {Gaussian, Bilateral}` × `ρ ∈ [1e-3, 0.3]` × `σ` (its range, or the noise-tied
default); `iter` fixed ≈ 50; start = ridge s2².

---

## 5. Summary — the Phase-A axes

| param | role | moves solution or speed? | predicted range | Phase-A values |
|---|---|---|---|---|
| `denoiser` | the implicit prior (shapes the null space) | **solution** | Gaussian / Bilateral (3D); TV Bregman ignores z | discrete, per truth |
| `sigma` | denoising level = prior strength at fixed ρ | **solution** | Gaussian [10,50], TV Bregman [2,15]; others not transferable | σ sweep per denoiser |
| `rho` | data-step ridge × prior strength (ρσ²) × speed | **solution** | [1e-3, 0.3] | {1e-3 … 3e-1} |
| `iter` | iterations to the fixed point | speed | [20, 50] | fixed ≈ 50 |
| `forced_pos`, `delta`, `init` | positivity / anisotropy / start | fixed | on / estimated / ridge s2² | — |

**Take-away:** ADMM-PnP = "which denoiser, how strong (σ), how much data (ρ)", with the prior's
overall strength the product **ρ·σ²**. Versus PnP: one knob fewer (no schedule), no
coarse-to-fine, and no cap on ρ (standard dual step). Versus ADMM: a free prior (any denoiser)
instead of sparsity only. On MA-TIRF use a 3D denoiser + ADMM-PnPv2's scale-free ÷peak + ridge
start; the default (adjoint start, σ = 15 unscaled, ρ = 1.0) sits just outside every range.
