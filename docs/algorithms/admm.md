# ADMM & ADMMv2 — a priori analysis of the parameters

Reads on top of `00_foundation.md`. ADMM (Alternating Direction Method of Multipliers) is a
**MAP** estimator with a **fixed** prior: sparsity (L1) + positivity. It does **not** use the
objective's `reg` / `lambda_reg` / κ (foundation §3.2) — its prior is hard-wired. ADMMv2 is a
separate solver with a cleaner threshold and dual step (§4). Physical operator throughout
(s1 = 38.9, s2 = 6.02, s3 = 0.562, s4 = 0.033). Math in plain text.

---

## 1. What ADMM solves — and its Bayesian reading

```
minimize over u, f :   ½·||H·u − g||²  +  τ·||f||₁  +  ι_{f ≥ 0}(f)     subject to  u = f
```

`u` is the copy "seen by the data", `f` the copy "seen by the prior", tied by `u = f`.

**Bayesian reading.** `½||Hu−g||²` is the Gaussian negative log-likelihood (up to the noise
scale); `τ·||f||₁` is a **Laplacian prior** `p(f) ∝ exp(−τ·||f||₁)` (i.i.d. Laplace on the
voxels = "most voxels are ~0, a few are bright"); `ι_{f≥0}` is a uniform prior on the positive
orthant. So ADMM is the MAP estimator for **a sparse, non-negative object under Gaussian
noise** — the right model for point-like / filamentary structures, the wrong one for a
continuous membrane.

Note it uses the **plain** `½||Hu−g||²`, not the scaled D of foundation §3.1: ADMM does not
carry the noise level, and τ is an absolute weight, not a share.

---

## 2. Logic, pseudo-code, and the fixed point (`solvers/admm.py`)

```
threshold  t = κ · max(g) / ||H|| ,   ||H|| = s1        (κ = threshold_ratio)
f ← solve_normal(Hᵀg, μ) = (HᵀH + μI)⁻¹ Hᵀg            # ridge start, from f = η = 0
η ← 0
repeat for `iter`:
    u ← (HᵀH + μI)⁻¹ ( Hᵀg + μ·(f − η) )               # data step: ridge fit pulled to f−η
    f ← max( u + η − t , 0 )                            # prox step: soft-threshold at t, then ≥0
    η ← η + μ·(u − f)                                   # dual step (note the factor μ)
```

### 2.1 The objective actually reached

At a fixed point (`u = f`, η stationary) the two steps' optimality conditions combine into the
KKT conditions of

```
min over f ≥ 0 :  ½·||Hf − g||²  +  τ·||f||₁ ,   with   τ = μ · t = μ · κ · max(g) / s1
```

**The prior's weight is a product of two knobs: `τ = μ · κ · max(g)/s1`.** In textbook ADMM the
threshold is τ/μ, so μ only sets speed; here the threshold t is fixed, so **μ changes the
solution** (§3.2). This is the awkwardness ADMMv2 removes.

### 2.2 The dual step and the stability cap on μ

The standard scaled dual update is `η ← η + (u − f)`. This code multiplies it by μ:
`η ← η + μ·(u − f)`. That is ADMM with a **relaxed dual step equal to μ**, which converges only
for a dual step in `(0, (1+√5)/2) ≈ (0, 1.618)` (Glowinski). So μ has **three roles at once**:
ridge weight of the data step, factor of the prior weight τ, and dual step size. The last one
caps it: **μ > 1.618 is expected to oscillate or diverge** (measured on a toy problem: μ = 2.5
→ NaN).

---

## 3. The parameters of ADMM v1

### 3.1 `threshold_ratio` (κ) — the sparsity threshold *(critical, but hard to predict)*

The soft-threshold zeroes every voxel of `u + η` below `t = κ·max(g)/s1`. Since `max(g)/s1` is
the scale of f's peak (foundation §1.2), naively `t ≈ κ·(peak of f)` — i.e. κ would be "the
threshold as a fraction of the peak", useful around 0.02–0.3.

**But this is exactly the parameter the a priori analysis cannot pin, and here is why.** The
data step `u` is a ridge reconstruction: with only 2–3 determined directions, it **spreads the
object's energy across the ~47 null-space depth directions** (foundation §1.1). So the voxels
of `u` are far smaller than a hypothetical concentrated peak — a diffuse object can have *every*
voxel below `κ·(peak)` for a modest κ, and the threshold then empties it. How much energy sits
above the threshold depends on the truth's shape, which is not known a priori.

- `κ → 0` : no sparsity → non-negative least squares; the null space is decided by μ's ridge
  and by `iter` (§3.3) → noisy / depth-collapsed.
- `κ → large` : threshold above the (spread) voxel values → empty image.
- **Predicted useful band: κ ∈ [0.01, 0.3]**, wider for concentrated truths (`vesicles`,
  `fibres`), narrow-to-none for the continuous `cell` (L1 is the wrong prior there). **This is
  the least certain prediction in the whole analysis** — it entangles κ with μ, with s1, and
  with the truth's spread — so Phase A must sweep it **log-spaced over a wide band**
  (`κ ∈ {3e-3, 1e-2, 3e-2, 1e-1, 3e-1}`) per truth, not around a single value.

### 3.2 `mu` (μ) — penalty *(critical, three coupled roles)*

1. **Data-step ridge.** `(HᵀH + μI)⁻¹` keeps directions with `s_i² ≫ μ` and damps the rest
   (foundation §4). `μ ≈ s3² = 0.3` keeps s1, s2 fully and s3 half; `μ ≪ s4² = 1e-3` lets noise
   through the poorly determined directions.
2. **Prior weight.** `τ = μ·t` (§2.1): at fixed κ, larger μ ⇒ sparser result.
3. **Dual step.** Must stay below ≈ 1.618 (§2.2).

**Predicted range: μ ∈ [s3², 1.6] ≈ [0.3, 1.6]** on MA-TIRF (the default 0.5 sits in it, ≈ s3²).
Because μ moves the solution *and* the speed *and* has a hard cap, it is genuinely
three-in-one — a design smell ADMMv2 fixes. **Phase A:** `μ ∈ {0.1, 0.3, 0.5, 1.0, 1.5}`; show
that it changes the sparsity (not only the speed) and that ≥ 1.618 oscillates.

### 3.3 `iter` — iterations *(critical only when κ ≈ 0)*

- With κ in band, the iterates converge to the lasso solution (§2.1); beyond a few tens they
  plateau → `iter` is a budget.
- With `κ ≈ 0`, each data step is one proximal (iterated-Tikhonov) step; after k steps the
  filter along a weak direction behaves like a ridge of weight ~μ/k. **Stopping early is then
  itself the regularization** → 20 vs 2000 iterations give different images.

**Predicted:** with κ in band the result plateaus by `iter ≳ 50`; with κ ≈ 0 it drifts. Since a
few tens are cheap, **Phase A should give ADMM a generous budget (≈ 200)** and read the plateau
from the `||u−f||/||f||` gap the code logs — this is exactly the setting a user flagged as
under-iterated at the default 20.

### 3.4 Not offered

No `init` (the start is the data step), no `reg` / `lambda_reg`, no noise model but Gaussian.

---

## 4. ADMMv2 — the principled threshold and dual step (`solvers/admm_v2.py`)

Same chain, two changes that remove the coupling of §2:

```
τ  = κ · max(Hᵀg)          (κ = kappa)      # absolute prior weight, independent of μ
threshold = τ / μ                            # so the fixed point's weight is τ, not μ·t
f ← solve_normal(Hᵀg, μ), clamped ≥ 0
η ← 0
repeat for `iter`:
    u ← (HᵀH + μI)⁻¹ ( Hᵀg + μ·(f − η) )
    f ← max( u + η − τ/μ , 0 )
    η ← η + (u − f)                          # standard dual step (no μ factor)
```

### 4.1 Why `κ` is now scale-free and interpretable

`τ_max = max(Hᵀg)` is **exactly** the weight that empties the image. Optimality of the lasso
at `f = 0` (with positivity): `f = 0` is the minimizer iff `Hᵀg ≤ τ` componentwise, i.e. iff
`τ ≥ max(Hᵀg)`. So with `τ = κ·max(Hᵀg)`:

```
κ ≥ 1  → empty image            κ → 0  → non-negative least squares
κ ∈ (0,1)  → the fraction of the "empties-it" weight that is applied
```

This reads the **same whatever the data scale, the operator gain, or the normalization of g** —
unlike v1's κ, which is `τ_v1 = μ·κ·max(g)/s1`, entangled with μ and s1 (for the same sparsity,
`κ_v1 ≈ κ_v2 · max(Hᵀg)·s1 / (μ·max(g))`, i.e. hundreds of times apart).

### 4.2 The dual step frees μ

With the standard `η ← η + (u−f)` the method converges for **any μ > 0**, and the fixed-point
weight is τ (independent of μ, §4.1). So in ADMMv2:

- **κ ∈ (0,1)** = the sparsity, scale-free, moves the **solution**;
- **μ = speed only** — no cap, no effect on the solution once converged; fastest when the
  data-step ridge conditions the determined directions well, `μ ∈ [s3², s1²] ≈ [0.3, 1.5e3]`;
- **iter** = a pure budget (default 100).

*Evidence (`solvers/_tests.py`, sparse toy):* ADMMv2 — κ = 1 returns exactly zero; μ = 0.3 and
μ = 30 reach the same solution to 5e-7. ADMM v1 — at a fixed threshold, μ from 0.3 to 1.7 moves
the solution from 21 to 6 non-zero voxels, μ = 2.5 → NaN.

**Phase A (ADMMv2):** `κ ∈ {0.02, 0.05, 0.1, 0.2, 0.5}` (the real solution axis, transferable to
deconvolution); `μ` fixed in [s3², s1²] (e.g. 1.0); `iter` a generous budget. Verify that any μ
gives the same reconstruction once converged, and that κ's best value transfers across truths /
noise / problems better than v1's `threshold_ratio`.

---

## 5. Summary — the Phase-A axes

| param | solver | role | moves solution or speed? | predicted range | Phase-A values |
|---|---|---|---|---|---|
| `threshold_ratio` κ | ADMM | sparsity threshold (entangled) | **solution** | [0.01, 0.3], uncertain | log band {3e-3 … 3e-1} per truth |
| `mu` μ | ADMM | ridge × prior weight × dual step | **solution** + speed, cap 1.618 | [0.3, 1.6] | {0.1, 0.3, 0.5, 1.0, 1.5} |
| `iter` | ADMM | budget; regularization if κ≈0 | speed (solution if κ≈0) | ≥ 200 | fixed generous |
| `kappa` κ | ADMMv2 | sparsity as fraction of τ_max | **solution**, scale-free | (0,1), useful [0.02, 0.5] | {0.02, .05, .1, .2, .5} |
| `mu` μ | ADMMv2 | speed only | speed | [s3², s1²] | fixed (e.g. 1.0) |
| `iter` | ADMMv2 | budget | speed | ≥ 100 | fixed generous |

**Take-away:** ADMM v1's `threshold_ratio` is the one parameter the a priori analysis can only
bracket loosely (it entangles the threshold with μ, the operator gain, and the truth's depth
spread), so it needs a **wide log sweep with a generous iteration budget** — the opposite of
the default (κ = 0.1, iter = 20). ADMMv2's `κ ∈ (0,1)` is the principled, scale-free version and
is the axis to trust for the sparse-prior family.
