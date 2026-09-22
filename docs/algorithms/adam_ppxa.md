# Adam & PPXA — a priori analysis of the parameters

Reads on top of `00_foundation.md` (the inverse problem, the shared objective L, the
spectrum, the scale f ~ 0.02). Adam and PPXA **minimize the same L** (foundation §3); they
differ only in *how* they descend it — which is why "do they return the same image?" tests
the framework rather than being a coincidence. Both are **MAP** estimators and both consult
`reg` / `lambda_reg` (unlike ADMM / PnP / MCMC). Ranges below are *predicted* from the maths;
Phase A tests them **around** these values.

Math is written in plain text (code blocks / Unicode) on purpose, so it stays readable
without a LaTeX renderer. Notation: `||x||` = L2 norm, `s1 s2 s3` = singular values of H,
`s_i^2` = eigenvalues of HᵀH, `f ~ 0.02` = f peaks around 0.02 on MA-TIRF.

---

## 0. The objective, restated

```
minimize over f ≥ 0 :   L(f) = (1−λ)·D(Hf,g)  +  λ·κ·R(f)          λ ∈ [0,1]
                         D    = ||Hf−g||² / (2·b·n_g)              (Gaussian noise, variance b)
                         κ    = 1 / R(f_init)                      (C2 calibration, foundation §3.2)
```

L is convex (foundation §3.4). So *in principle* the minimizer is unique and both solvers
must reach it; *in practice* the near-flat valley (foundation §1.1) means that unless the
`λ·κ·R` term weighs on the null space, the finite-iteration result depends on the path. This
is the recurring split below:

> **A parameter moves the *solution* when the prior is weak, and only the *convergence* when
> the prior weighs.**

---

## 1. The shared (objective) parameters — *what* is minimized

### 1.1 `reg` — the prior *(critical: it decides the ~47 undetermined directions)*

The six priors and their homogeneity are in foundation §3.3. The MA-TIRF-specific reasoning:

- the data fix only span(v1, v2, [v3]); **`reg` alone shapes the depth profile** of every
  column (foundation §1.1). This is the most consequential choice, well ahead of the optimizer.
- degree-1 priors (L1, TV, Hessian-Frobenius, SHV) grow linearly with the image scale;
  degree-2 (L2, Tikhonov) grow like the square, so at `f ~ 0.02` they are intrinsically ~50×
  weaker for the same nominal λ — expect degree-2 priors to need a larger λ and to over-smooth
  once they bite.

**Predicted:** for the synthetic truths (sparse objects with smooth axial profiles), TV / SHV
/ L1 beat L2 / Tikhonov; for a continuous membrane (`cell`), the Hessian-based priors
(Hessian-Frobenius, SHV) give the smoothest surface. **Phase A:** treat `reg` as a discrete
axis — sweep λ (below) *per prior*, compare priors at each one's best λ.

### 1.2 `lambda_reg` (λ) — the prior's weight, now a share *(critical)*

At an interior minimum the gradient balance is

```
(1−λ)·∇D  =  −λ·κ·∇R
```

Because `κ = 1/R(f_init)` puts R on D's scale (foundation §3.2), λ reads as a **regularization
share**, not a dead knob:

- `λ → 0` : non-negative least squares. The 47 undetermined directions are set by noise
  amplification along s3, s4 and by the path → *predicted*: noisy / depth-collapsed (realism
  check: "no better than ridge", "collapsed").
- `λ → 1` : the data are dropped; f → argmin R (= 0 for L1/L2/SHV, a constant for TV/Tikhonov)
  → realism check's "empty".
- in between : the trade-off, spread across [0,1] instead of crammed near 1 thanks to κ. The
  D1 study measured the best NMSE near `λ ≈ 0.05–0.1` on the calibrated form.

**Predicted useful range: λ ∈ [0.02, 0.4]**, prior-dependent (degree-2 priors toward the top).
**Phase A:** sweep `λ ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5}` per prior; report the share
`r(λ) = 1 − R(f_λ)/R(f_{λ=0})`, the NMSE, and the realism verdict along it.

### 1.3 `rho` — SHV's sparse/smooth balance *(critical, SHV only)*

```
R_SHV(f) = mean_x sqrt( ρ²·||Hess f(x)||_F²  +  (1−ρ)²·f(x)² )
```

`ρ → 1` : pure Hessian-Frobenius (smooth, not sparse). `ρ → 0` : pure L1 (sparse, not smooth).
**Predicted:** sparse truths `ρ ≈ 0.3–0.6`, continuous membrane `ρ → 0.8–1`. **Phase A:**
`ρ ∈ {0.1, 0.3, 0.6, 0.9}` per truth (only when `reg = SHV`).

### 1.4 `delta` (δ) — anisotropy Δz/Δxy *(fixed, not swept)*

Weights the axial derivative so a gradient means the same along z and laterally; a property of
the grid and optics. **Always the estimated value** (the operator provides it). Only the
derivative-based priors (Tikhonov, TV, Hessian, SHV) use it. Not a Phase-A axis.

### 1.5 noise model — L2 for MA-TIRF at moderate noise; PPXA is Gaussian-only

D's b is the measurement's, set in `[noise-model]`, not by the solver. PPXA accepts only the
Gaussian model (its data step is a least-squares solve); Adam accepts all three. Not an atlas
axis (a separate Poisson experiment covers it).

---

## 2. Adam — logic and the parameters of the descent

### 2.1 Logic and pseudo-code (`solvers/adam.py`)

Gradient descent on L with a per-coordinate adaptive step (Adam), positivity by clamping, a
learning rate halved whenever the loss rises, and a small-change stop.

```
f ← f0                                   # the ridge s2^2 start (foundation §4)
init Adam optimizer over f with step size lr
repeat for max_iter:
    loss ← L(f)
    gradient ← ∇ loss           (autograd)
    f ← Adam_step(f, gradient, lr)       # f ← f − lr · m̂ / (√v̂ + ε)
    f ← max(f, 0)                        # positivity, by projection
    every K iterations:
        if loss rose since last check:      lr ← lr / 2      # step overshot
        elif loss fell by less than EPS:    stop             # converged
```

### 2.2 The mathematics of the step

Adam keeps running means m (gradient) and v (gradient²) and moves each voxel by

```
Δf = − lr · m̂ / (√v̂ + ε),      with   | m̂ / √v̂ | ≈ 1
```

The ratio is **dimensionless and O(1)** by construction: **each voxel moves by about `lr` per
iteration, whatever the gradient's magnitude.** That is the key to reading `lr` — it is a
*displacement in the units of f* (~0.02), not a gradient scaling.

Convexity gives a unique minimizer when the prior weighs (foundation §3.4). The loss is **not
monotone** under Adam, so the scheduler halves lr only when the loss *rises*, and the stop
tests `|Δloss| < EPS`.

### 2.3 `lr` — learning rate *(critical only if too small)*

A step in units of f, with `max f ≈ 0.02` (foundation §1.2):

- `lr ≫ max f` (the default 0.1 is ~5× too large): first steps overshoot, the loss rises, the
  scheduler halves lr until it fits — a few wasted checks, then normal descent.
  **Self-correcting.**
- `lr ≪ max f` : each voxel crawls; within the budget f barely leaves its start → realism
  check's "trivial". **Not self-correcting** (the scheduler only ever decreases lr).

Solution vs convergence: with a weighing prior, lr sets *speed* only (all lr in range reach
the same image); with a weak prior it also picks the null-space point the finite run stops at.

**Predicted: any `lr ∈ [max f0, 10·max f0]` is equivalent; below `0.1·max f0` it fails.** Rule:
*start too large, never too small.* **Phase A:** `lr ∈ {0.5, 1, 2, 5} × max f0` on one truth,
to confirm the plateau and the low-end failure (a diagnostic, not a per-truth axis).

### 2.4 `init` / `lambda_rr` — where the descent starts *(critical for Adam, through the scale)*

For a convex L the start does not change the minimizer, but it changes **how many steps** Adam
needs (§2.2: it walks in steps of lr) and **which null-space point** a finite run reaches. The
three starts, at MA-TIRF's scale (f ~ 0.02):

| start | peak | vs f | why |
|---|---|---|---|
| `adjoint`  Hᵀg | ~20 | ~850× too large | scale ~ s1²·f |
| `ridge`, λ_rr = 1e4 | ~0.0017 | ~13× too small | ≈ Hᵀg / 1e4 : a rescaled back-projection |
| `ridge`, λ_rr = s2² ≈ 36 | ~f | right scale | keeps s1 fully, s2 half |

From `adjoint`, Adam must walk every voxel from ~20 down to ~0.02 in steps of lr while the
scheduler keeps halving lr on the oscillations — the budget runs out first (the v1 "depth
shift" regression). **Predicted:** (i) `ridge` with `λ_rr ∈ [s2², s1²]` converges fastest and
is the default; (ii) `adjoint` fails within Adam's budget on MA-TIRF; (iii) with a strong
prior and a long run all starts agree. **Phase A:** `init ∈ {ridge(s2²), adjoint}` on one
truth × two λ (a diagnostic of the claim, not a per-truth axis).

### 2.5 `max_iter`, `K`, `EPS` — budget and stopping *(fixed)*

All three are **fixed**, not swept:

- `K = 10` — the loss is checked every 10 iterations. Adam uses the full gradient (no
  stochasticity), so 10 is fine.
- `EPS = 1e-8` — an **absolute** threshold on `|Δloss|`. With `L ~ 1e-2 … 1e-1`, `1e-8` means
  "7–8 digits": the run stops when the loss has essentially stopped moving.
- `max_iter = 3000` (at least) — large enough that **the EPS stop, not the budget, ends the
  run**. The ill-conditioning (foundation §1.1) slows descent, so a small budget would stop
  mid-valley (path-dependent); 3000 lets the loss plateau and EPS fire. Most runs stop before
  3000; a run that hits it did not converge, which is itself informative.

---

## 3. PPXA — logic and the parameters of the splitting

### 3.1 Logic and pseudo-code (`solvers/ppxa.py`)

PPXA (Combettes–Pesquet 2008) minimizes a **sum of convex terms** by evaluating each one's
proximal operator in parallel, averaging, and relaxing. Three terms with equal weights 1/3:
f1 = data, f2 = prior, f3 = positivity indicator.

```
f ← f0 ;  u_i ← f   (i = 1,2,3)
repeat for max_iter:
    p1 ← prox_data(u1) = (I + γ·HᵀH)⁻¹ (u1 + γ·Hᵀg)     # pull toward the data
    p2 ← prox_reg(u2, γ/w)                               # prox of (γ/w)·λ·R
    p3 ← max(u3, 0)                                      # prox of the positivity indicator
    p̄  ← (p1 + p2 + p3) / 3
    u_i ← u_i + relax·(2·p̄ − f − p_i)                    # i = 1,2,3
    f   ← f + relax·(p̄ − f)
```

PPXA runs on L rescaled by `w = quadratic_weight` so its data term is exactly `½·||Hf−g||²` (a
pure rescaling; the minimizer is unchanged), which is why the prior prox weight is `γ·λ/w`.

### 3.2 The mathematics of `gamma` (γ)

The data prox moves u toward the data, direction by direction:

```
p1 = (I + γ·HᵀH)⁻¹ (u + γ·Hᵀg)

along singular direction i :   [p1]_i = (1/(1+γ·s_i²))·u_i  +  (γ·s_i²/(1+γ·s_i²))·[LS]_i
```

So `γ·s_i²` is the **gain** on direction i: `γ ≫ 1/s_i²` pulls direction i fully to the data,
`γ ≪ 1/s_i²` barely touches it. To handle the determined directions s1, s2 without overreaching,
γ must bracket their inverse squares:

```
γ ∈ [ 1/s1² , 1/s2² ] ≈ [ 7e-4 , 0.03 ]   on MA-TIRF
```

- `γ ≪ 1/s1²` : even s1 barely moves → very slow.
- `γ ≫ 1/s2²` : each prox jumps to its own term's minimizer; averaging three conflicting
  minimizers oscillates, *and* the prior's iterative prox (TV: 30 inner iterations,
  Hessian/SHV: 20) stops being exact → PPXA can **stall above** the true minimum.

### 3.3 Convergence, and `lambda_relax`

Theory: PPXA converges to a minimizer for **any γ > 0** and any relaxation `ρ_n ∈ (0,2)` with
`Σ ρ_n·(2−ρ_n) = ∞` — so γ and the relaxation change *speed only*, **provided the proxes are
exact**. The §3.2 caveat is exactly when they are not. `lambda_relax` is held fixed in (0,2)
(default 1.5; > 1 over-relaxes = faster when proxes are exact, → 2 is the stability edge).
Predicted best in [1.0, 1.5]; not a per-truth axis.

**Phase A:** `γ ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}` (spanning the bracket and just outside),
MA-TIRF and deconvolution; confirm the plateau inside the bracket, the slow-down below, the
stall above with a non-smooth prior (TV/SHV).

### 3.4 `init`, `max_iter`, `K`, `EPS` *(fixed)*

Same as Adam: `K = 10`, `EPS = 1e-8`, `max_iter = 3000` (at least) so the EPS stop ends the
run, not the budget. PPXA is exact in the limit, so with enough iterations the start matters
less than for Adam — **but only if γ is in range** (otherwise it stalls regardless, and hits
the budget). `init` = ridge s2².

---

## 4. Adam vs PPXA — what "the same solution" means

| situation | same L? | same image? |
|---|---|---|
| strong prior, both converged | yes | **yes** (unique minimizer) |
| weak prior (flat valley) | yes, to ~1e-3 | **no** — each stops at its own null-space point |
| γ out of range / non-smooth prior with inexact prox | no — PPXA stalls higher | no |

The framework's claim is precise: **with γ in its bracket and a prior that weighs, Adam and
PPXA return the same image.** A disagreement is a diagnostic — the prior is too weak, or a step
size is wrong.

---

## 5. Summary — the Phase-A axes

| param | solver | role | moves solution or speed? | predicted range | Phase-A values |
|---|---|---|---|---|---|
| `reg` | both | the prior (decides the null space) | **solution** | 6 priors | all six, compared at best λ |
| `lambda_reg` | both | prior share (calibrated) | **solution** | [0.02, 0.4] | 0, .02, .05, .1, .2, .35, .5 (per prior) |
| `rho` | both (SHV) | sparse ↔ smooth | **solution** | 0.3–0.6 sparse, →1 continuous | .1, .3, .6, .9 |
| `delta` | both | anisotropy | fixed | estimated | — |
| `lr` | Adam | step in units of f | speed (solution if prior weak) | [max f0, 10·max f0] | {.5,1,2,5}×max f0 (diagnostic) |
| `init` | Adam | start scale | speed (solution if prior weak) | ridge s2² | ridge vs adjoint (diagnostic) |
| `gamma` | PPXA | prox gain per direction | speed (+correctness if non-smooth) | [1/s1², 1/s2²] | 1e-3 … 1e-1 |
| `lambda_relax` | PPXA | relaxation | speed | [1, 1.5] | fixed 1.5 |
| `max_iter`,`K`,`EPS` | both | budget / stop | speed | K=10, EPS=1e-8, max_iter=3000 | fixed |

**The atlas's real axes for Adam/PPXA are `reg` × `lambda_reg` (× `rho` for SHV)** — the ones
that move the solution. `lr`, `init`, `gamma`, `lambda_relax` are convergence knobs, tested
once as diagnostics (do they behave as predicted?) rather than per truth.
