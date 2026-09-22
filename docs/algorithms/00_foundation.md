# Foundation — the inverse problem, the Bayesian objective, and the numbers every solver inherits

> Status: **a priori theoretical analysis** (redone from the source code, not the earlier
> notes). This file is the shared ground the per-algorithm notes build on: the problem, the
> estimator, the objective every solver minimizes, the operator's spectrum, and the scale of
> the unknown. Every parameter range in the other notes is derived from the quantities fixed
> here. Convention, fixed for MA-TIRF: **`normalize` (the operator flag) is `False`** — H is
> the physical operator, never divided by its largest singular value.

Math is written in plain text (code blocks / Unicode) so it stays readable without a LaTeX
renderer. Notation: `||x||` = L2 norm, `s1 s2 s3 …` = singular values of H (largest first),
`s_i^2` = eigenvalues of HᵀH, `f ~ 0.02` = f peaks around 0.02.

---

## 1. The inverse problem

```
g = H·f + n ,        f ≥ 0
```

- `f` — the unknown object (MA-TIRF: a 3D volume, nz depth planes × lateral pixels;
  deconvolution: a 2D image);
- `H` — the **forward operator** (linear, known): MA-TIRF mixes the nz depth planes into
  n_angles measured stacks *independently in each lateral pixel*, so H is one small
  (n_angles × nz) matrix applied to every depth column; deconvolution convolves with a known
  PSF;
- `n` — noise set by the detector (Gaussian read noise of variance b, Poisson shot noise, or
  both — `core/noise.py`); g is peak-normalized, so `max g = 1`.

`f ≥ 0` (an intensity) is a hard constraint, enforced by every solver.

### 1.1 What makes it hard — the spectrum of H (MA-TIRF, physical operator)

SVD of the per-column matrix: `H = U·Σ·Vᵀ`, `Σ = diag(s1 ≥ s2 ≥ …)`. For the synthetic
microscope (13 angles, 62.6–69.8°, nz = 50 planes over 0–300 nm), from
`operator.singular_values()` with `normalize = False`:

| s1 | s2 | s3 | s4 | s5 | s6… |
|---|---|---|---|---|---|
| 38.9 | 6.02 | 0.562 | 0.033 | 1.4e-3 | < 5e-5 |

The gaps are enormous: `s1/s2 ≈ 6.5`, `s2/s3 ≈ 11`, `s3/s4 ≈ 17`. In squares (the eigenvalues
of HᵀH, which every ridge / step-size rule compares against):

```
s1² ≈ 1.5e3 ,   s2² ≈ 36 ,   s3² ≈ 0.32 ,   s4² ≈ 1.1e-3
```

**Only 2–3 depth directions per column are determined by the data**; the other ~47 of 50 lie
in the numerical null space of H. This one fact drives everything:

1. **The prior decides the null space.** The data fix f only in span(v1, v2, [v3]); the
   remaining ~47 directions are set by the regularizer and by positivity alone. *The choice of
   prior and its weight is the single most important decision*, far more than the optimizer.
2. **A near-flat valley.** D is nearly constant along the null space, so many f share almost
   the same L. An optimizer stopped early lands somewhere in that valley depending on its path
   (start, step, iterations) — *implicit regularization*. A well-posed use is one where the
   prior, not the path, picks the point.
3. **A huge condition number.** The data term's curvature spans `s1²/s3² ≈ 5e3` (and
   `s1²/s4² ≈ 1e6`): step sizes and ridge weights must be read against this spectrum, never
   against 1.

For **deconvolution** H is a convolution: its singular values are `|ĥ(ω)|`, a *continuous*
spectrum decaying smoothly to 0 at high frequency — ill-posed too, but without MA-TIRF's clean
"2–3 directions then nothing". Ranges tied to discrete `s_i²` therefore do **not** transfer
as-is between the two problems.

### 1.2 The scale of the unknown

g is peak-normalized (`max g = 1`). Along the best-determined direction, `g1 = s1·f1`, so
`f1 ≈ 1/s1 ≈ 1/38.9 ≈ 0.026`. Hence

```
f peaks around 0.02–0.03 on MA-TIRF.
```

(Measured on the benchmark truths: the f that explains g peaks at 0.022–0.025.) **Every
parameter expressed "in units of f" — a learning rate, a proposal noise level, a sparsity
threshold — must be compared to ~0.02, not to 1.** This is the most common way a default
calibrated for a photograph (peak 1) goes wrong on MA-TIRF (peak 0.02): by a factor
`≈ s1 ≈ 40`. For deconvolution f peaks near 1, so those same defaults are in range.

---

## 2. The Bayesian view: MAP and MMSE

Posterior: `p(f|g) ∝ p(g|f)·p(f)` (Bayes). Two estimators:

- **MAP** — the mode: `f̂ = argmax p(f|g) = argmin [ −log p(g|f) − log p(f) ]`. Every solver
  here except MCMC is a MAP estimator: it *minimizes an energy*.
- **MMSE** — the posterior mean `f̂ = E[f|g]`, the minimizer of the expected squared error.
  Intractable in closed form (a high-dimensional integral); MCMC approximates it by sampling
  and averaging. Smoother than MAP, and it reports an acceptance rate that says whether the
  chain explored.

With a Gibbs prior `p(f) ∝ exp(−Φ(f))` and the noise likelihood, the MAP problem is
`min over f≥0 of  D_θ(Hf,g) + Φ(f)`, where `θ = (a,b)` is the noise level. This is §3.

---

## 3. The objective every MAP solver shares (`core/objective.py`)

```
L(f) = (1−λ)·D(Hf,g)  +  λ·κ·R(f) ,      f ≥ 0 ,   λ ∈ [0,1]
```

### 3.1 The data term D carries the noise level, and sits at ≈ ½

D is the negative log-likelihood **per pixel** (`solvers/fidelities/base.py`):
`D(Hf,g) = −(1/n_g)·log p(g|f) + const`, so `D(g,g) = 0`. For Gaussian noise of variance b:

```
D(Hf,g) = ||Hf − g||² / (2·b·n_g)
```

Three properties, all used later:

- **It carries the noise level b**: a noisier measurement is trusted less *by construction*,
  not by asking λ to compensate.
- **It is a mean** over the n_g pixels of g: it does not grow with image size.
- **Chi-square anchor.** At the truth, `Hf* = g − n`, so `||Hf*−g||² = ||n||²`, `E = n_g·b`,
  giving `E[D(f*)] = ½` — **whatever the problem, size or noise**. So D = O(1), anchored at ½
  (good fit), 0 (perfect fit). `chi2_ratio = 2·D` for the Gaussian model.

### 3.2 The prior term R, and the calibration κ

R is a **mean over voxels** (`solvers/regularizers/`), one of six priors (§3.3). It has **no
intrinsic scale**: on MA-TIRF, where `f ~ 0.02`, R(f) is 1e-2…1e-3 — two to three orders below
`D ~ ½`. Raw, λ would be a dead knob until within 1e-3 of 1.

The framework fixes this with a single scalar `κ = 1/R(f_init)` (`reg_scale`, set by
`pipeline.build_objective`; f_init = the ridge s2² start): it puts R on D's scale so λ reads
as a **regularization share** (decision D1 = B/C2). The reported
`regularization_share  r(λ) = 1 − R(f_λ)/R(f_{λ=0})` is invariant under κ (it cancels), so the
ruler is untouched.

> **Note.** ADMM, PnP and ADMM-PnP do **not** use this objective's R / λ: their prior is
> hard-wired (soft-threshold, or a denoiser). λ, κ and R concern only Adam, PPXA and the MCMC
> acceptance term — see each note.

### 3.3 The six regularizers and their homogeneity

| R | favours | degree p in R(c·f) = c^p·R(f) |
|---|---|---|
| L1 norm | few bright voxels (sparse) | 1 |
| L2 norm | small values overall | 2 |
| L2 of the gradient (Tikhonov) | smooth, blurs edges | 2 |
| L1 of the gradient (TV) | piecewise-constant, sharp edges | 1 |
| Frobenius of the Hessian | piecewise-linear, no staircase | 1 |
| Sparse Hessian Variation (SHV) | sparse **and** smooth jointly | 1 |

The degree p matters: a degree-2 prior grows like the *square* of the image scale, so it
reacts differently to the `f ~ 0.02` scale than a degree-1 prior.

### 3.4 Convexity and uniqueness

D is convex; all six R are convex; {f ≥ 0} is convex → L is convex, so a global minimum exists
and any correct solver reaches the same L. It is **strictly** convex only if R is strictly
convex on the null space of H (§1.1). Consequence used throughout:

- **prior weighs enough** → the minimizer is essentially unique → all solvers/starts agree on
  the *image*, and every parameter then affects only *convergence* (speed);
- **prior too weak** (small λ, or a prior blind to the null space) → a flat valley → the
  *image* depends on the path (start, steps, iterations), and parameters change the *solution*,
  not just the speed.

The lens for every parameter: does it move the solution, or only the convergence?

---

## 4. Conventions used in every note

- Operator: **physical** (`normalize = False`), spectrum `s1 = 38.9, s2 = 6.02, s3 = 0.562,
  s4 = 0.033`; `||H|| = s1`, `||HᵀH|| = s1² ≈ 1.5e3` (the `lipschitz` value).
- g peak-normalized (`max g = 1`); f peaks `≈ 0.02` on MA-TIRF, `≈ 1` on deconvolution.
- "Determined directions": s1, s2, and s3 partly (`s3² ≈ 0.3`); everything below s4 is
  noise-dominated.
- A ridge / penalty weight `w` acts as a threshold on `s_i²`: directions with `s_i² ≫ w` are
  kept / fit, those with `s_i² ≪ w` are cut / frozen. This one mechanism appears as
  Adam/PPXA's ridge start, ADMM's μ, PnP's α, ADMM-PnP's ρ and MCMC's λ_rr.
