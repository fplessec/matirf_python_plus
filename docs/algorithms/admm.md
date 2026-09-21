# ADMM — theory of the parameters, before any benchmark

> Status: **theoretical note** (Day 7, step 2). Ranges are *predicted* from the mathematics
> and from the MA-TIRF spectrum (see `adam_ppxa.md` §1.1: $s_1 = 38.9$, $s_2 = 6.02$,
> $s_3 = 0.562$, $s_4 = 0.033$). No algorithm was modified; §4 lists *proposals*.

---

## 1. What this ADMM solves

ADMM (Alternating Direction Method of Multipliers) splits a problem into two easy halves
tied by a constraint. Here the unknown is duplicated as $u$ (seen by the data) and $f$ (seen
by the prior), with $u = f$:

$$
\min_{u,f}\; \tfrac12 \lVert Hu - g \rVert_2^2 \;+\; \tau \lVert f \rVert_1 \;+\; \iota_{f \ge 0}(f)
\quad \text{s.t.} \quad u = f .
$$

The prior is **fixed**: sparsity ($\ell_1$) plus positivity. ADMM does not use the
objective's `reg` / `lambda_reg`, nor the noise level: its data term is the plain
$\tfrac12\lVert Hu-g\rVert_2^2$ (Gaussian noise only).

One iteration of the code, with $\eta$ the (scaled) dual variable:

| step | code | meaning |
|---|---|---|
| data | $u = (H^TH + \mu I)^{-1}\big(H^Tg + \mu (f - \eta)\big)$ | a ridge-regularized fit to the data, pulled toward $f - \eta$ |
| prior | $f = \max(u + \eta - t,\ 0)$ | soft threshold at $t$, then positivity |
| dual | $\eta = \eta + \mu\,(u - f)$ | accumulate the disagreement between $u$ and $f$ |

with the threshold $t = \kappa\, \max(g) / \lVert H \rVert$ ($\kappa$ = `threshold_ratio`,
$\lVert H\rVert = s_1$ from a power iteration). The run starts from the data step with
$f = \eta = 0$, i.e. from the ridge estimate $(H^TH + \mu I)^{-1} H^T g$.

### 1.1 The objective actually reached

At a fixed point ($u = f$), the optimality conditions of the two steps combine into

$$
H^T(g - Hf) \in \mu\, t\; \partial \lVert f \rVert_1 + N_{f\ge 0}(f),
$$

which are exactly those of $\min_{f\ge0} \tfrac12\lVert Hf-g\rVert^2 + \tau\lVert f\rVert_1$ with

$$
\boxed{\tau = \mu \cdot t = \mu\, \kappa\, \max(g) / s_1 .}
$$

**The prior's weight is the product of two parameters.** In textbook ADMM the threshold is
$\tau/\mu$, so that $\mu$ only affects speed; here the threshold is fixed and $\mu$ therefore
changes the solution itself (see §4).

### 1.2 The dual step

In the standard scaled form the dual update is $\eta \leftarrow \eta + (u - f)$. The code
multiplies it by $\mu$: this is ADMM with a *relaxed dual step* equal to $\mu$, which
converges for dual steps in $(0,\ \tfrac{1+\sqrt5}{2}) \approx (0, 1.618)$ (Glowinski). So $\mu$
has **three roles at once**: ridge weight of the data step, factor of the prior weight, and
dual step size. The last one caps it: **$\mu > 1.618$ is expected to oscillate or diverge.**

---

## 2. The parameters

### `threshold_ratio` $\kappa$ — the sparsity threshold *(critical)*

Since $f \sim g / s_1$ (`adam_ppxa.md` §1.2), $t = \kappa \max(g)/s_1 \approx \kappa \max f$:
**$\kappa$ is the threshold as a fraction of the image's peak**. At each iteration, voxels
whose value (after the data step) falls below about $\kappa \times$ the peak are set to zero.

- $\kappa \to 0$: no sparsity; ADMM tends to non-negative least squares — ill-posed on
  MA-TIRF (47 undetermined depth directions per column), so the result is then decided by
  the data step's ridge and by the number of iterations (see `iter`).
- $\kappa \to 1$ (threshold ≈ peak): everything is thresholded away → empty image.
- In between: sparsity. **Predicted useful range: $\kappa \in [0.02, 0.3]$** — larger for
  sparse truths (`vesicles`, `fibres`), and a poor fit whatever $\kappa$ for continuous
  objects (the `cell` membrane), since $\ell_1$ on voxels is the wrong prior for them.

This is ADMM's only real "what does the object look like" knob — the counterpart of
Adam/PPXA's `reg` + `lambda_reg`, restricted to one prior (sparsity).

### `mu` $\mu$ — penalty *(critical, but mostly fixed by a rule)*

1. **Data step.** $(H^TH + \mu I)^{-1}$ keeps the singular directions with $s_i^2 \gg \mu$ and
   damps the others — the same mechanism as the ridge weight $\lambda_{rr}$ of MCMC. On
   MA-TIRF: $\mu \ll s_2^2 = 36$ keeps the two determined directions each iteration;
   $\mu \approx s_3^2 = 0.32$ also half-keeps the third; $\mu \ll s_4^2 = 10^{-3}$ lets noise
   through along the poorly determined directions.
2. **Prior weight** $\tau = \mu t$ (§1.1): at fixed $\kappa$, a larger $\mu$ means a sparser
   result.
3. **Dual step** (§1.2): must stay below ≈ 1.618.

**Predicted range on MA-TIRF: $\mu \in [s_3^2,\ 1.6] \approx [0.3, 1.6]$.** The default 0.5 sits
in it (and $\approx s_3^2$). For deconvolution (normalized PSF, $s_1 = 1$, a continuous
spectrum) the data step is a Wiener filter of parameter $\mu$; $\mu$ should then be of the order
of the noise-to-signal power, **$\mu \in [10^{-4}, 10^{-1}]$**, well inside the dual limit.

### `iter` — number of iterations *(critical when $\kappa$ is small)*

ADMM starts from the ridge estimate and converges toward the $\ell_1$-regularized solution.

- With $\kappa > 0$ well chosen: the iterates converge; beyond a few tens of iterations
  nothing changes (a plateau) → `iter` is a budget.
- With $\kappa \approx 0$: each data step is a proximal (iterated-Tikhonov) step; after $k$
  iterations the filter along direction $i$ behaves like a ridge of parameter $\sim \mu/k$.
  **Stopping early is then itself the regularization**: 20 iterations and 2 000 do not
  give the same image. This is v1's behaviour with its default `iter = 20`.

**Prediction:** with $\kappa \in [0.02, 0.3]$ the result is stable for `iter` $\gtrsim 50$;
with $\kappa \to 0$ it drifts with `iter` toward a noisier image.

### Not offered, on purpose

No `init` (the start is the data step itself), no `reg`/`lambda_reg`, no noise model other
than Gaussian.

---

## 3. Summary

| parameter | role | critical? | predicted range (MA-TIRF) | toward 0 | toward large |
|---|---|---|---|---|---|
| `threshold_ratio` $\kappa$ | sparsity threshold, ≈ fraction of the peak | **yes** | 0.02 – 0.3 | NNLS, decided by `iter` | empty image |
| `mu` $\mu$ | data-step ridge × prior weight × dual step | yes, by rule | 0.3 – 1.6 | noise through poorly determined directions | data barely used; > 1.618: oscillation |
| `iter` | budget, or implicit regularization if $\kappa \approx 0$ | only if $\kappa \approx 0$ | ≥ 50 | the ridge start | converged ($\kappa>0$) / noisier ($\kappa\approx0$) |

For a user, ADMM comes down to **one decision: how sparse ($\kappa$)** — provided $\mu$ is in
its range. It is the natural method for sparse objects and the wrong one for continuous
ones.

---

## 4. Proposals (not applied — for the project owner to decide)

1. **Decouple $\mu$ from the prior weight.** Apply the threshold $\tau/\mu$ with
   $\tau = \kappa \max(g)/s_1$ fixed. Then $\kappa$ alone sets the prior and $\mu$ only the speed,
   as in textbook ADMM. *Evidence*: §1.1 (derivation); to be confirmed by H-A2.
2. **Standard dual step** $\eta \leftarrow \eta + (u - f)$, removing the 1.618 cap on $\mu$ and
   letting $\mu$ be chosen from the spectrum alone ($[s_3^2, s_1^2]$). *Evidence*: §1.2; to be
   confirmed by H-A2 (oscillation above 1.618 today).

Both would change results of existing configurations; neither is needed for the benchmark,
which characterizes ADMM as it is.

---

## 5. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-A1 | $\kappa \to 0$: NNLS-like (noisy / depth errors); $\kappa \gtrsim 0.5$: empty; best $\kappa \in [0.02, 0.3]$ on sparse truths; poor on `cell` for any $\kappa$ | $\kappa$ sweep per truth, realism check |
| H-A2 | at fixed $\kappa$, $\mu$ changes the sparsity (not only the speed); $\mu > 1.618$ oscillates | $\mu$ sweep 0.03 – 3, loss / iterate history |
| H-A3 | with $\kappa$ in range the result plateaus for `iter` ≳ 50; with $\kappa \approx 0$ it drifts with `iter` | `iter` sweep 10 – 2 000 at two $\kappa$ |
| H-A4 | $\mu \approx s_3^2$ – 1 converges fastest on MA-TIRF | iterations to plateau vs $\mu$ |
