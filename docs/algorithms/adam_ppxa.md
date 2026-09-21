# Adam and PPXA — theory of the parameters, before any benchmark

> Status: **theoretical note** (Day 7, step 2). Every range below is *predicted* from the
> mathematics and from the MA-TIRF operator's spectrum; the benchmark exists to confirm or
> refute each prediction. Numbers marked *(pre-study)* come from the exploratory runs of
> 2026-09-21 on one cropped truth and are not yet benchmark results.

Adam and PPXA are treated together because they minimize **the same objective**. They
differ only in *how* they descend it — which is exactly what makes their comparison a test
of the framework: with the right parameters, they must find the same reconstruction.

---

## 1. The objective they share

$$
\min_{f \ge 0}\; L(f) = (1-\lambda)\, D(Hf, g) + \lambda\, R(f), \qquad \lambda \in [0, 1]
$$

- $D$ — the data fidelity: the per-pixel negative log-likelihood of the noise model,
  scaled by the noise level (Gaussian: $D = \frac{1}{2 b n_g}\lVert Hf-g\rVert_2^2$, $b=\sigma^2$).
- $R$ — the regularization (the prior), a **mean over voxels**.
- $f \ge 0$ — positivity, enforced by projection (Adam) or as a third proximal term (PPXA).

Every term is convex (D, all six regularizations, the positivity constraint), so $L$ is
convex. It is **strictly** convex only if $R$ is strictly convex on the directions $H$ does
not see — this single fact explains most of what follows (§1.3).

### 1.1 The MA-TIRF operator, in numbers

With the synthetic microscope (13 angles, 62.6–69.8°, 50 planes over 0–300 nm), each depth
column is acted on by a $13 \times 50$ matrix whose singular values are

| $s_1$ | $s_2$ | $s_3$ | $s_4$ | $s_5$ | $s_6 \dots s_{13}$ |
|---|---|---|---|---|---|
| 38.9 | 6.02 | 0.562 | 0.0331 | 0.00137 | $< 5\cdot10^{-5}$ |

So **only 2–3 directions per column are determined by the data**; 47 of the 50 depth
degrees of freedom are fixed by the prior (and positivity) alone. Three consequences used
throughout:

1. the reconstruction is dominated by the prior → the choice of $R$ and $\lambda$ is the
   single most important decision;
2. the objective has a **near-flat valley** (the quasi-null space of $H$) → an optimizer
   stopped before convergence lands somewhere in that valley depending on its *path*
   (initialization, step size, iterations): *implicit regularization*;
3. the curvature of the data term spans $s_1^2/s_3^2 \approx 5\,000$ → step sizes must be
   chosen relative to this spectrum (PPXA, §3.2).

### 1.2 Scale of the unknown

`g` is normalized (peak = 1). Since $Hf \approx g$ and $s_1 \approx 39$, the reconstruction
has values $f \sim g / s_1 \sim$ **0.01–0.03** (measured: the f that explains g peaks at 0.022–0.025 on the benchmark truths). Every parameter measured in "units of f"
(Adam's learning rate, the MCMC noise level) must be compared to this scale, not to 1.

### 1.3 The flat valley and uniqueness

If $R$ is weak (small $\lambda$, or a prior of small magnitude), many $f$ reach almost the
same $L$. Then:

- two different algorithms, or the same one from two starts, **can return very different
  images with the same $L$** — observed *(pre-study)*: Adam and PPXA equal to $10^{-3}$ in
  $L$, yet their images are far apart (angle 0.5–0.8) at $\lambda = 0.5$;
- the answer depends on the path, not on the objective — which is not a reconstruction
  method one can reason about.

When $R$ weighs enough to fix the null space, the minimizer becomes (nearly) unique and all
paths converge to it. **A well-posed use of Adam/PPXA is one where the prior, not the path,
decides.** This is the lens for every parameter below.

---

## 2. The shared (objective) parameters

These describe *what* is minimized; they are identical for Adam and PPXA.

### `reg` — the regularization *(critical)*

The prior belief about the object. Each option favours a kind of image:

| Regularization | Favours | Homogeneity |
|---|---|---|
| L1 norm | few bright voxels (sparse) | degree 1 |
| L2 norm | small values overall (mildest) | degree 2 |
| L2 norm of the gradient (Tikhonov) | smooth everywhere, blurs edges | degree 2 |
| L1 norm of the gradient (TV) | piecewise-constant, sharp edges | degree 1 |
| Frobenius norm of the Hessian | piecewise-linear, no staircasing | degree 1 |
| Sparse Hessian Variation (SHV) | sparse **and** smooth, jointly | degree 1 |

"Degree" is the homogeneity: $R(cf) = c^p R(f)$. It matters for $\lambda$ (below): a degree-2
prior grows like the *square* of the image scale, a degree-1 prior linearly.

**Prediction.** For the synthetic truths (sparse objects with smooth profiles), SHV, TV and
L1 should beat L2 and Tikhonov; for the membrane (`cell`), Hessian-type priors should give
the smoothest surfaces.

### `lambda_reg` — the prior's weight *(critical)*

- $\lambda \to 0$: non-negative least squares. On MA-TIRF, the 47 undetermined depth
  directions are then set by the path and by noise amplification along $s_3, s_4$ →
  expected: noisy, possibly depth-collapsed reconstructions (the realism check's
  "collapsed" / "no better than ridge").
- $\lambda \to 1$: the data are ignored; $f \to \arg\min R$ ($f = 0$ for L1, L2, SHV;
  a constant for TV, Tikhonov) → the realism check's "empty".
- In between: the trade-off.

**Why $\lambda$ is not (yet) a percentage.** With the current formulation the balance at the
solution is set by $\lambda/(1-\lambda)$ times $D$'s curvature over $R$'s scale, and $R$'s scale
follows $f$'s ($\sim 0.01$, raised to the degree $p$). So $R \ll D$: $R$ weighs almost nothing
until $\lambda$ is very close to 1. *(pre-study)*: with TV, the regularization share
$r = 1 - R(f_\lambda)/R(f_0)$ stays at 0.01 / 0.07 / 0.17 for $\lambda$ = 0.1 / 0.5 / 0.9, and
the best reconstruction needs $\lambda \ge 0.99$. **Predicted useful range today:
$1-\lambda \in [10^{-4}, 10^{-1}]$, i.e. $\lambda \in [0.9, 0.9999]$, prior-dependent.**
Making $\lambda$ interpretable is decision D1 (normalizing $D$ and $R$ by reference values),
studied separately.

### `rho` — SHV's trade-off *(critical, SHV only)*

$R(f) = \text{mean}_x \sqrt{\rho^2 \lVert \mathrm{Hess}\, f(x) \rVert_F^2 + (1-\rho)^2 f(x)^2}$

- $\rho \to 1$: pure Hessian-Frobenius (smooth, not sparse);
- $\rho \to 0$: pure L1 (sparse, not smooth);
- SPITFIRe's default 0.6 — "moderately sparse".

> Note for the internship report (III.4): its formula is right, but one sentence inverts the
> limits ("si fixé à 0 … seulement les variations hessiennes"). $\rho = 0$ is pure sparsity.
> Its three levels (0.9 / 0.6 / 0.1 = weakly / moderately / strongly sparse) are consistent
> with the formula.

**Prediction.** Sparse truths (`vesicles`, `fibres`) prefer $\rho \approx 0.3$–0.6; the
continuous membrane prefers $\rho \to 0.8$–1.

### `delta` — anisotropy ratio *(not tuned: fixed by the physics)*

The axial derivative is weighted by $\delta = \Delta z / \Delta xy$ so that a gradient means the
same thing along z and in the plane. It is a property of the grid and the optics, which is
why the GUI **estimates** it. **Recommendation: always use the estimated value**; it is a
parameter that must be *kept fixed*, not explored. Only regularizations with derivatives
use it (Tikhonov, TV, Hessian, SHV).

### The noise model (`[noise-model]` section)

PPXA accepts only the Gaussian model (its data step is a least-squares solve). Adam accepts
all three. Not critical for MA-TIRF at moderate noise: L2 is expected to be enough (to be
checked in the Poisson experiment).

---

## 3. Solver parameters

### 3.1 Adam

Adam keeps running averages of the gradient ($m$) and of its square ($v$), and moves each
voxel by

$$ \Delta f = -\,\mathrm{lr}\; \frac{\hat m}{\sqrt{\hat v} + \epsilon}. $$

The ratio $\hat m / \sqrt{\hat v}$ is **dimensionless and of order 1**: each voxel moves by about
`lr` per iteration, *whatever the scale of the gradient*. Hence:

#### `lr` — learning rate *(critical only if too small)*

It is a step **in units of f**. With $f \sim 0.01$–0.03 (§1.2):

- `lr` $\gg \max f$ (the default 0.1 is ~5× too large): the first steps overshoot, the loss
  rises, and the built-in scheduler halves `lr` every K iterations until it fits — a few
  wasted checks, then normal descent. **Self-correcting.**
- `lr` $\ll \max f$: each voxel crawls; within the iteration budget the image barely leaves
  its starting point → the realism check's "trivial". **Not self-correcting** (the
  scheduler only ever *decreases* `lr`).

**Prediction: any `lr` in $[\max f_0,\ 10 \max f_0]$ gives the same result; below
$0.1 \max f_0$ it fails.** The asymmetry is the practical rule: *start too large, never too
small*. (A future improvement would express `lr` relative to $\max f_0$ so the rule holds for
any data scale — a proposal, not a change.)

#### `init` and `lambda_rr` — where the descent starts *(critical for Adam, through the scale)*

For a convex objective the start does not change the minimizer — but it changes **which
point of the flat valley** a finite run reaches (§1.3) and, for Adam, **how many steps** it
needs, since each voxel moves by about `lr` per iteration. Measured on the benchmark truths
(the f that explains g peaks at 0.022–0.025):

| start | peak value | relative to f | shape |
|---|---|---|---|
| `adjoint`, $H^T g$ | ~20 | **~850× too large** (order $s_1^2$) | back-projection |
| `ridge`, $\lambda_{rr} = 10^4$ | ~0.0017 | ~13× too small | $pprox H^Tg / 10^4$: the same back-projection, rescaled |
| `ridge`, $\lambda_{rr} \in [s_2, s_1]$ | ~f | right scale | contains the two determined components |

Hence the v1 regression explained *(measured: cosine 0.11 from `adjoint` vs 0.89 from
`ridge(1e4)`, same 5 000 iterations)*: from `adjoint`, Adam must walk every voxel down from
~20 to ~0.02 in steps of `lr`, and the scheduler halves `lr` each time the descent
oscillates — the budget runs out on the way. From `ridge(1e4)` it only has to climb a
factor ~13 from nearly the right shape. The benefit of v1's ridge start was therefore
mostly its **scale**, not its shape (with $\lambda_{rr} = 10^4 \gg s_1^2$, it *is* a scaled
back-projection).

**Prediction:** (i) with a strong enough prior and enough iterations, all starts converge to
the same image; (ii) with Adam's budget, `adjoint` fails on MA-TIRF whatever the prior;
(iii) `ridge` with $\lambda_{rr} \in [s_2, s_1]$ converges fastest. Recommendation: `ridge`,
fixed — and a scale-free alternative would be to rescale any start to match $\lVert g
Vert$
(a proposal, not a change). For PPXA the start matters far less: its data prox rescales
toward the data within a few iterations.

#### `max_iter`, `K`, `EPS` — budget and stopping *(comfort)*

- `max_iter`: budget. Too small → a path-dependent answer (§1.3); never harmful when large
  thanks to `EPS`.
- `K`: the loss is checked every K iterations; if it rose, `lr` is halved; if it fell by
  less than `EPS`, the run stops. Adam here uses the full gradient (deterministic), so
  there is no stochastic noise to average: any K in 10–50 behaves the same.
- `EPS`: an **absolute** threshold on the loss change. With $L \sim 10^{-2}$–$10^{-1}$,
  $10^{-8}$–$10^{-10}$ means "converged to 7–8 digits". (It is scale-dependent: another
  normalization of $D$ would shift its meaning.)

### 3.2 PPXA

PPXA (Combettes & Pesquet, 2008) minimizes a sum of convex functions by evaluating each one's
**proximal operator in parallel**, averaging, and relaxing. Here three terms:
$\;f_1 = $ data, $f_2 = $ prior, $f_3 = $ positivity indicator, with equal weights
$\omega_i = 1/3$:

$$
p_i = \mathrm{prox}_{\gamma f_i/\omega_i}(u_i), \quad
\bar p = \sum_i \omega_i p_i, \quad
u_i \leftarrow u_i + \rho_n (2\bar p - f - p_i), \quad
f \leftarrow f + \rho_n (\bar p - f).
$$

Convergence to a minimizer is guaranteed **for any $\gamma > 0$** and relaxations
$\rho_n \in (0, 2)$ with $\sum \rho_n (2-\rho_n) = \infty$. So $\gamma$ and the relaxation change
*speed*, never the answer — provided the proximal operators are exact.

(Implementation note: PPXA works on $L$ rescaled so that its data term is
$\frac12 \lVert Hf - g\rVert_2^2$ — a pure rescaling, the minimizer is unchanged.)

#### `gamma` — the proximal step *(critical in practice)*

The data prox is $(I + \gamma H^TH)^{-1}(u + \gamma H^T g)$: along singular direction $i$ it
moves $u$ toward the data by the factor $\gamma s_i^2 / (1 + \gamma s_i^2)$.

- $\gamma \ll 1/s_1^2$ ($\approx 7\cdot10^{-4}$): even the best-determined direction barely moves
  → very slow.
- $\gamma \gg 1/s_3^2$ ($\approx 3$): every prox jumps to its own term's minimizer; averaging
  three conflicting minimizers oscillates → slow again. Worse, the prior's prox weight grows
  with $\gamma$, and the iterative proxes (TV: 30, Hessian/SHV: 20 inner iterations) stop
  being exact → PPXA can stall above the true minimum.
- In between, directions $s_1, s_2$ are handled well and the prior fills the rest.

**Prediction: $\gamma \in [1/s_1^2,\ 1/s_2^2] \approx [7\cdot10^{-4},\ 0.03]$ on MA-TIRF.**
*(pre-study)*: $\gamma = 0.01$ reached Adam's $L$ exactly; $\gamma \ge 1$ failed with TV.
The default 0.05 sits just above the range. For deconvolution (normalized PSF, $s_1 = 1$),
the same rule predicts $\gamma \sim 1$.

#### `lambda_relax` — relaxation *(keep fixed)*

Theory: any constant in $(0, 2)$ converges; $> 1$ over-relaxes (faster when the proxes are
exact), $\to 2$ is the stability edge. **The relaxation is now fixed** (default 1.5; v1 halved
it whenever the loss rose — see `solvers/ppxa.py` for why that was dropped: a geometric decay
breaks the convergence condition $\sum \rho_n(2-\rho_n) = \infty$, and PPXA's loss is not
monotone, so the halving fired on normal transients). *(measured, 1500 iterations)*: fixed
1.5 never let the loss rise and ended within 1–2 % of the best $L$; fixed 1.9 reached the
same $L$ but could oscillate; the halving ended up to 7 % higher. **Recommendation: 1.0–1.5.**

#### `init`, `lambda_rr`, `max_iter`, `K`, `EPS`

As for Adam. PPXA is exact in the limit, so with enough iterations the start matters less
than for Adam — but only if $\gamma$ is in range.

---

## 4. Adam versus PPXA — what "the same solution" can mean

| Situation | Same $L$? | Same image? |
|---|---|---|
| strong prior, both converged | yes | **yes** (unique minimizer) |
| weak prior (flat valley) | yes, to $\sim10^{-3}$ | **no** — each stops at its own point of the valley |
| $\gamma$ out of range, or non-smooth prior with inexact prox | no — PPXA stalls higher | no |

So the benchmark's claim is precise: **with $\gamma$ in its range and a prior that weighs,
PPXA and Adam return the same reconstruction.** A disagreement is then a diagnostic —
either the prior is too weak for the reconstruction to be defined by the objective, or a
step size is wrong.

---

## 5. Critical versus comfort parameters

| | Critical (changes the reconstruction) | Fixed by rule | Comfort |
|---|---|---|---|
| shared | `reg`, `lambda_reg`, `rho` (SHV) | `delta` (estimated), noise model (L2) | — |
| Adam | — (`lr` only if too small) | `init = ridge`, `lr` $\in [\max f_0, 10\max f_0]$ | `max_iter`, `K`, `EPS` |
| PPXA | `gamma` (speed, and correctness with non-smooth priors) | `lambda_relax` fixed $\in [1, 1.5]$, `init = ridge` | `max_iter`, `K`, `EPS` |

For a user, Adam and PPXA therefore come down to **three decisions: which prior, how much,
and (SHV) how sparse** — everything else follows from the data or can be left at its rule.

---

## 6. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H1 | Useful $\lambda$ lies in $[0.9, 0.9999]$ today, prior-dependent | $\lambda$ sweep, log-spaced in $1-\lambda$, all priors |
| H2 | $\lambda \to 0$ collapses or matches ridge; $\lambda \to 1$ empties | realism check along the sweep |
| H3 | Adam: same result for `lr` $\in [\max f_0, 10\max f_0]$, failure below $0.1\max f_0$ | `lr` sweep |
| H4 | Adam from `adjoint` fails within budget on MA-TIRF; `ridge` in $[s_2, s_1]$ converges fastest; with a strong prior and a long run, all starts agree | init × $\lambda$ × iterations |
| H5 | PPXA: $\gamma \in [1/s_1^2, 1/s_2^2]$; failure far outside | $\gamma$ sweep, MA-TIRF and deconvolution |
| H6 | a fixed `lambda_relax` $\in [1, 1.5]$ is as good as 1.9 and safer *(partly shown, see §3.2)* | relaxation sweep, TV/SHV |
| H7 | Adam = PPXA (image) when H5 holds and the prior weighs | paired runs, angle between images |
| H8 | SHV: $\rho \approx 0.3$–0.6 for sparse truths, $\to 1$ for the membrane | $\rho$ sweep per truth |
| H9 | SHV / TV / L1 beat L2 / Tikhonov on these truths | comparison at each prior's best $\lambda$ |
| H10 | L2 is enough for MA-TIRF; Poisson fidelity helps only at low photon counts | Poisson noise, Adam with L2 vs KL vs PG |
