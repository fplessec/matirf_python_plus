# ADMM-PnP (Plug-and-Play ADMM) — theory of the parameters, before any benchmark

> Status: **theoretical note** (Day 7, step 2). MA-TIRF spectrum as in `adam_ppxa.md` §1.1
> ($s_1 = 38.9$, $s_2 = 6.02$, $s_3 = 0.562$, $s_4 = 0.033$). The denoisers, their scale
> problem and their MA-TIRF limits are analysed once in `pnp.md` §2–3 and not repeated here.
> ADMM-PnP is benchmarked on MA-TIRF only. ADMM-PnP itself is unchanged; its proposed
> fixes exist as the separate solver ADMM-PnPv2 (§4).

---

## 1. What ADMM-PnP does

The augmented-Lagrangian counterpart of PnP (Venkatakrishnan et al., 2013; Chan et al.,
2017). ADMM splits $\min \tfrac12\lVert Hx - g\rVert^2 + \Phi(v)$, $x = v$, and replaces the
proximal step of the (unknown) prior $\Phi$ by a denoiser:

| step | code | meaning |
|---|---|---|
| data | $x = (H^TH + \rho I)^{-1}\big(H^Tg + \rho(v - u)\big)$ | fit the data, pulled toward $v - u$ |
| prior | $v = D_\sigma(x + u)$, then $v \leftarrow \max(v, 0)$ | denoise at a **fixed** level $\sigma$ |
| dual | $u = u + (x - v)$ | accumulate the disagreement between data and prior |

The run returns $v$ (the denoised variable). The data term is the plain
$\tfrac12\lVert Hx-g\rVert^2$ (Gaussian noise only); neither `reg`/`lambda_reg` nor the noise
level is used. Start: $x = v = f_0 = H^Tg$ (`adjoint`, not configurable), $u = 0$.

### 1.1 Compared with ADMM and PnP

- **Dual step: standard.** Unlike this project's ADMM (`admm.md` §1.2), the update is
  $u \mathrel{+}= (x - v)$: no hidden coupling, no $1.618$ cap on $\rho$.
- **No schedule.** PnP (HQS) needs Zhang's annealing ($\alpha \nearrow$, $\sigma \searrow$) because it
  has no dual variable; here the dual $u$ carries the accumulated mismatch, so $\rho$ and $\sigma$
  can stay fixed. Two knobs instead of three, but no coarse-to-fine progression.
- **Data step, direction by direction** (as `pnp.md` §1.1 with $\alpha \to \rho$): directions with
  $s_i^2 \gg \rho$ follow the data, those with $s_i^2 \ll \rho$ keep $v - u$. On MA-TIRF the depth
  structure is again essentially the denoiser's.

### 1.2 What $\rho$ and $\sigma$ mean together

If the denoiser were the exact MAP denoiser of a prior $\lambda\,\Phi$ under Gaussian noise of level
$\sigma$, its step would be $\mathrm{prox}_{\lambda\Phi\,\sigma^2}$; ADMM needs $\mathrm{prox}_{\Phi/\rho}$. So

$$ \lambda_{\text{implicit}} \;\propto\; \rho\,\sigma^2 : $$

**the prior's strength is the product of the two**. Raising $\rho$ at fixed $\sigma$ strengthens the
prior *and* trusts fewer directions of the data; raising $\sigma$ at fixed $\rho$ strengthens the
prior only. Unlike textbook (convex) ADMM, where $\rho$ changes only the speed, **here $\rho$ changes
the solution** — PnP-ADMM has no fixed objective, its fixed point depends on $(\rho, \sigma)$.

### 1.3 Convergence

Not guaranteed for an arbitrary denoiser with fixed $\rho$: known results need a non-expansive
(or bounded) denoiser, or an increasing $\rho$ (Chan et al., 2017). In practice the iterates
either settle (a plateau) or oscillate slightly; the benchmark measures which.

---

## 2. The parameters

### `denoiser` — the prior *(critical)*

Same catalogue and same MA-TIRF caveats as PnP (`pnp.md` §2): TV Bregman works slice by slice
(ignores z); Gaussian is a no-op for $\sigma \le 7.5$; Bilateral, Wiener and DCT depend on the
image's intensity scale (`pnp.md` §3), which on MA-TIRF (peak ≈ 0.02) makes a given $\sigma$ far
stronger than on a photograph. DCT is considered broken and not benchmarked.

### `sigma` — the denoising level *(critical)*

Fixed for every iteration (default 15, 0–255 scale). With §1.2, it is the prior's strength at
fixed $\rho$.

- $\sigma \to 0$: no denoising → no prior → the undetermined directions keep the start
  ($H^Tg$, badly scaled, §4).
- $\sigma$ large: over-smoothing; structures erased, and the data only restore the 2–3
  determined directions.

**Predicted useful range:** Gaussian $\sigma \in [10, 50]$ (the default 15 = a 0.6 px kernel),
TV Bregman $[2, 15]$; Bilateral / Wiener at MA-TIRF's scale: far below their deconvolution
values (not transferable — `pnp.md` §3).

### `rho` — the penalty *(critical)*

Three effects at once (§1.1–1.2): the ridge of the data step (which directions follow the data),
the prior strength $\rho\sigma^2$, and the speed.

- $\rho \to 0$: the data dominate; noise enters along $s_3, s_4$; the prior barely acts.
- $\rho \gg s_2^2 = 36$: even the determined directions follow the denoiser; the data are
  nearly ignored.

**Predicted range on MA-TIRF: $\rho \in [s_4^2, s_3^2] \approx [10^{-3}, 0.3]$** (trust $s_1, s_2$ and
$s_3$ partly) — the same logic, and the same range, as PnP's final weight $\lambda_{kz}$. The
default 1.0 is just above it (it already hands $s_3$ to the denoiser). For a normalized PSF
(deconvolution), a Wiener-like $\rho$ of the order of the noise-to-signal power.

### `iter` — number of iterations *(comfort, within limits)*

With fixed $(\rho, \sigma)$ the iterates approach a fixed point or oscillate around it (§1.3).
**Predicted: a plateau within 20–50 iterations; beyond, either nothing changes or a slight
oscillation.** The default 20 is at the low end.

### `forced_pos` — positivity *(keep fixed: on)*

Clamps $v$ after denoising. Keep `True`.

### `delta` — anisotropy for Gaussian and Bilateral *(fixed by the physics)*

As for PnP: it should be the estimated anisotropy ratio, but defaults to 1 and is not
estimated automatically for denoisers (§5).

### Not offered: `init`

As for PnP: the start is $H^Tg$, ~850× too large on MA-TIRF. Here it enters twice — as $x$, and
as $v$ in the first data step, where every undetermined direction keeps it (§1.1).

---

## 3. Summary

| parameter | role | critical? | predicted range (MA-TIRF) | toward small | toward large |
|---|---|---|---|---|---|
| `denoiser` | the prior | **yes** | Gaussian / Bilateral (3D); TV Bregman ignores z | — | — |
| `sigma` | denoising level (prior strength at fixed $\rho$) | **yes** | Gaussian 10–50, TV Bregman 2–15; others not transferable | no prior | structures erased |
| `rho` | data-step ridge × prior strength × speed | **yes** | $10^{-3}$ – 0.3 | noise along $s_3, s_4$ | data ignored |
| `iter` | iterations to the fixed point | comfort | 20 – 50 | not converged | plateau / slight oscillation |
| `forced_pos` | positivity | fixed: on | — | — | — |
| `delta` | axial kernel width | fixed: estimated | — | — | — |

For a user, ADMM-PnP comes down to **three decisions — which denoiser, how strong ($\sigma$), how
much to trust the data ($\rho$) — with the product $\rho\sigma^2$ as the prior's overall strength.**
Against PnP: one knob fewer (no schedule), no coarse-to-fine; against ADMM: a free prior (any
denoiser) instead of sparsity only.

---

## 4. ADMM-PnPv2 — the proposals, as a separate solver

At the project owner's request the proposals exist as the solver "ADMM-PnPv2"
(`solvers/pnp_admm_v2.py`); ADMM-PnP is unchanged. The benchmark compares them.

| | ADMM-PnP | ADMM-PnPv2 |
|---|---|---|
| denoiser input | $255(x+u)$ — assumes $f \in [0,1]$ | $255(x+u)/p$, $p$ = the start's peak — **scale-free** |
| `sigma` | fixed, default 15 | fixed; **empty = 3 × the measurement's noise level** (relative to its peak, 0–255) |
| `delta` | 1 by default | the operator's estimate when empty |
| start | $H^Tg$ (~850× too large on MA-TIRF) | ridge, $\lambda_{rr} = s_2^2$ by default (or `adjoint`) |
| positivity | switchable | always on |
| default `rho`, `iter` | 1.0, 20 | 0.1, 50 (inside the predicted ranges) |

The factor 3 tying $\sigma$ to the noise is a **first guess**: the benchmark settles it
(H-AP7). Not changed — open point shared with PNPv2: $\rho$ is still compared to the
eigenvalues of $H^TH$, so it does not transfer across operators of different gain.

*Evidence (`solvers/_tests.py`)*: on a 2D blur, multiplying $H$'s gain by 50 (with $\rho$
scaled by $50^2$) leaves ADMM-PnPv2's reconstruction exactly divided by 50 (gap
$9\cdot10^{-7}$).

## 5. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-AP1 | Gaussian / Bilateral (3D) beat TV Bregman (slice by slice) on depth structure | denoiser comparison |
| H-AP2 | $\rho$ changes the solution, not only the speed; best $\rho \in [10^{-3}, 0.3]$ on MA-TIRF | $\rho$ sweep at fixed $\sigma$ |
| H-AP3 | configurations with the same $\rho\sigma^2$ give similar reconstructions (Gaussian denoiser) | paired $(\rho, \sigma)$ along $\rho\sigma^2$ = const |
| H-AP4 | the iterates plateau within 20–50 iterations (or oscillate slightly) | `iter` sweep, change between consecutive iterates |
| H-AP5 | PnP (annealed) vs ADMM-PnP (dual, fixed): which is better on MA-TIRF, and which is easier to tune | comparison at each one's best settings, sensitivity to its knobs |
| H-AP6 | as for PnP: $\sigma$ transfers between problems only for Gaussian / TV Bregman | same $\sigma$ on MA-TIRF and deconvolution (denoiser-level test) |
| H-AP7 | ADMM-PnPv2 ≥ ADMM-PnP on MA-TIRF; its noise-tied $\sigma$ (3 × noise) is near the best $\sigma$ at every noise level | ADMM-PnP vs v2; $\sigma$ sweep per noise level |
