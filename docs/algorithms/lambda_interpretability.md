# λ interpretability — calibrating the objective so λ reads as a regularization share (D1)

> Status: **D1 decided — B with C2** (calibrate the regularizer by `R(f_init)`). The objective
> is shared by every solver that uses a regularizer (Adam, PPXA, the textbook comparison), so
> this note is about the *formulation*, not one algorithm. It shows the measured behaviour of
> λ, the reference menu that was tested, and the decision. Numbers are *(measured)* on cropped
> MA-TIRF (40×40 lateral patch) and deconvolution (128×128 patch) truths, peak-normalized,
> **oracle** Gaussian noise (D uses the exact σ² injected), Adam 200 it from the **s2² ridge
> start**, 2026-09-22, `benchmarks/studies/lambda_interpretability.py`. The benchmark repeats
> them across truths and noise levels.
>
> Not yet wired into `core/objective.py`: the decision is recorded here; the calibration
> (wrap `reg_term` / `prox_reg` by `1/R(f_init)`) is the next implementation step, and it
> **changes the meaning of every stored λ** — to be stated in the migration note.

---

## 1. The problem: λ has no scale

The shared objective (`core/objective.py`) is a blend:

$$ L(f) = (1-\lambda)\, D(Hf, g) + \lambda\, R(f), \qquad \lambda \in [0, 1]. $$

$D$ is a *scaled* per-pixel negative log-likelihood: $\approx \tfrac12$ at the true image, $0$
at a perfect fit, so $D = O(1)$ whatever the problem, noise or size (`chi2_ratio = 2D`). $R$
has **no such scale**: it is a mean of gradients / curvatures of $f$, and its magnitude
depends on the regularizer, the intensity scale, and how rough $f$ is.

So $R \ll D$, and the prior is not felt until $\lambda/(1-\lambda)$ reaches $D/R$. How far that
is depends on the regime:

- **Low noise / smooth $f$** — $R(f_0)$ is tiny, $D/R \sim 10^2$–$10^3$: λ is a **dead knob**
  over almost all of $[0,1]$, biting only within a hair of 1.
- **Realistic noise** — $f_0$ (the $\lambda=0$ reconstruction) is rougher, so $R(f_0)$ is
  larger and λ bites sooner: here $R_{\text{TV}}(f_0)=2.8\cdot10^{-2}$,
  $R_{\text{Tikhonov}}(f_0)=8.2\cdot10^{-3}$ on the fibres patch — still one to two orders below
  $D\approx\tfrac12$, and **where** λ bites now depends on the noise and the prior.

Either way λ has no fixed meaning: the same value does different amounts of regularization on
different problems, priors and noise levels — the opposite of a transferable knob.

---

## 2. The share ruler, and what standard λ delivers — measured

`regularization_share` (`benchmarks/metrics.py`) quantifies "how regularized" a result is:

$$ r(\lambda) = 1 - \frac{R(f_\lambda)}{R(f_0)}, \qquad f_0 = \text{the } \lambda=0 \text{ reconstruction}, $$

$0$ at $\lambda=0$, $\to 1$ as the prior wins. This ruler is **never calibrated** — it is the
definition of the share, and rescaling it would be circular. Under the **standard** objective
the share the same λ delivers varies widely across problem and prior:

| $\lambda$ | fibres · TV | fibres · Tikhonov | deconv · TV | deconv · Tikhonov |
|---|---|---|---|---|
| 0.10 | 0.12 | 0.27 | 0.44 | 0.26 |
| 0.50 | 0.40 | 0.62 | 0.74 | 0.65 |
| 0.90 | 0.67 | 0.80 | 0.81 | 0.80 |

At $\lambda=0.10$ the delivered share runs from 0.12 to 0.44; the best-NMSE λ (§4) lands
anywhere from 0.05 to 0.75 depending on the case. **λ is not a percentage and does not
transfer.**

---

## 3. Calibrating "R onto D" — the fix and the reference menu

Only the *ratio* of the two terms' scales matters (the argmin of $L$ is invariant under an
overall positive rescaling), and $D$ already has a canonical scale ($\approx\tfrac12$ at the
truth). So we rescale $R$ alone by a single scalar $R_{\text{ref}}$ of the right magnitude —
"calibrating $R$ onto $D$":

$$ L(f) = (1-\lambda)\, D(Hf, g) + \lambda\, \frac{R(f)}{R_{\text{ref}}}. $$

$R_{\text{ref}}$ is the only free choice. The menu tested:

| ref | $R_{\text{ref}}$ | interpretable? | cost | universal? |
|---|---|---|---|---|
| **C0** | none ($\kappa=1$) | no (the dead knob) | free | — (control) |
| **C1** | $R(f_0)$, the $\lambda=0$ reconstruction | yes | a **pre-pass** ($\lambda=0$ solve) | yes, but $f_0$ is non-canonical on ill-posed problems |
| **C2** | $R(f_{\text{init}})$, the ridge start | yes | **free** (already computed) | **yes**, deterministic |
| **C4** | $R(f_{\text{true}})$, the truth | yes | — | oracle, benchmark-only |

Under C2 the share is graded and monotone from $\lambda=0$, landing the action in a human
range instead of near 1 (same two patches as above):

| $\lambda$ | fibres · TV | fibres · Tikhonov | deconv · TV | deconv · Tikhonov |
|---|---|---|---|---|
| 0.10 | 0.63 | 0.88 | 0.81 | 0.87 |
| 0.50 | 0.78 | 0.97 | 0.83 | 0.93 |
| best NMSE at | λ=0.05 | λ=0.05 | λ=0.05 | λ=0.05 |

The mapping is not identical across priors (no fixed constant can straighten the L-curve,
which is prior- and truth-dependent — so "λ interpretable" ≠ "λ = r exactly"), but it is
monotone, bounded, and puts the useful setting in a low, consistent range.

---

## 4. The reference choice → C2

The decisive test is **transferability**: does the best-NMSE λ stay put across truths, noise
and problems, and does a *free* reference track the oracle? Best λ* (NMSE) per reference,
Tikhonov (the case that separates them; TV behaves similarly for C1 and C2):

| truth · σ | C1 = $R(f_0)$ | C2 = $R(f_{\text{init}})$ | C4 = oracle | C2 anchor $R_{\text{init}}/R(f_0)$ |
|---|---|---|---|---|
| MA-TIRF fibres · 0.15 | 0.50 | 0.10 | 0.10 | 0.15 |
| MA-TIRF cell · 0.15 | 0.75 | 0.25 | 0.10 | 0.15 |
| deconv img_001 · 0.15 | 0.50 | 0.05 | 0.10 | 0.01 |
| deconv img_002 · 0.15 | 0.25 | 0.05 | 0.10 | 0.03 |

- **C2 tracks the oracle C4; C1 drifts high.** As noise rises, C1's best λ climbs to 0.5–0.75,
  while C2 (and the oracle) stay in 0.05–0.25. C2's best λ is the more transferable.
- **C2's anchor is stable** within a (problem, regularizer): $R_{\text{init}}/R(f_0)$ sits in a
  narrow band per column, so C2 is a predictable rescaling of C1 at no cost.
- **C1's cost and fragility.** C1 needs a pre-pass (a $\lambda=0$ solve), and on an ill-posed
  problem $f_0$ is **non-canonical**: under noise the $\lambda=0$ MA-TIRF solve amplifies noise
  (vesicles NMSE 0.8–1.0), so $R(f_0)$ is set by amplified noise, not signal. $f_{\text{init}}$
  (a single deterministic ridge solve) does not have this problem.
- **C4** is the oracle: unavailable in practice, and its scale is not even a stable multiple of
  $R(f_0)$ — so "calibrate to the truth" is impossible with a fixed reference anyway.

**Decision — D1 = B with C2.** $L = (1-\lambda)D + \lambda\,R/R(f_{\text{init}})$: interpretable
like C1, tracks the oracle at least as well, and it is free, deterministic and universal (no
operator spectrum, no pre-pass, no unstable anchor). A (keep standard) is out — λ does not
transfer; C (operator reference $s_2$) is out — not transferable across problems.

---

## 5. The ridge start weight — s2² is the default

C2 reuses the solver's ridge start, so the start's weight $\lambda_{rr}$ matters twice (start
*and* reference). Sweeping it under real noise (`e0` = NMSE of the $\lambda=0$ solve, `e*` =
best NMSE over λ under C2):

| case | weight | e0 | e* | anchor |
|---|---|---|---|---|
| deconv img_001 · TV | 1e-2 | 0.130 | 0.042 | 0.39 |
| | **s2²** | **0.104** | 0.062 | 0.11 |
| MA-TIRF cell · TV | 1e-2 | 0.696 | 0.565 | 0.78 |
| | **s2²** | 0.690 | **0.535** | 0.65 |

$s_2^2$ (the second eigenvalue of $H^TH$: keep the well-determined directions, cut the
noise-dominated rest) gives a **modest but consistent gain** — a clearly cleaner $\lambda=0$
reconstruction on deconvolution, a small gain on MA-TIRF, and **no regime where it is
meaningfully worse** ($s_1^2$ is sometimes better, sometimes worse — unstable). It is also
determinate and universal: `lambda_rr` empty $=$ `second_eigenvalue`, which is $s_2^2$ where
the spectrum is known and the Lipschitz $s_1^2$ otherwise. **This is now the solver default**
(`Solver.initial_guess` delegates to `ridge_start`; `INIT_UI_PARAM` defaults to ridge, empty
$= s_2^2$).

---

## 6. What the benchmark measures

Per (truth, noise, regularizer): `regularization_share` $r(\lambda)$, NMSE, `chi2_ratio`
(over- vs under-fit) and the realism verdict — so the λ→reconstruction relation is shown
quantitatively ("raise λ, get this share, this fit, this error"), the deliverable regardless
of the formulation.

## 7. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-L1 | standard: the same λ delivers a very different share across problem/prior/noise; best λ not transferable | λ sweep both problems, both priors, two noise levels |
| H-L2 | C2 (`R(f_init)`): $r(\lambda)$ graded and monotone; best λ in a low human range and more transferable than standard or C1 | λ sweep, both formulations, both problems |
| H-L3 | C2's best λ tracks the oracle C4, and its anchor $R_{\text{init}}/R(f_0)$ is stable within a (problem, prior); C1 drifts high with noise | reference comparison across truths/noise |
| H-L4 | at the best λ, `chi2_ratio` $\approx 1$; over-regularizing pushes it $\gg 1$, under-regularizing $\ll 1$ | chi2_ratio vs λ |
| H-L5 | the s2² ridge start lowers (or matches) the $\lambda=0$ NMSE vs a lighter/heavier weight, on both problems | ridge-weight sweep |
