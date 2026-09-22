# λ interpretability — is the blend weight readable, and should the objective be calibrated? (D1)

> Status: **study + a decision to make (D1)**. The objective is shared by every solver that
> uses a regularizer (Adam, PPXA, and the textbook comparison), so this note is about the
> formulation itself, not one algorithm. It presents the measured behaviour of λ and lays out
> the decision **D1 — whether to calibrate the regularization term** — which is the project
> owner's to make. Numbers below are *(measured)* on one cropped MA-TIRF truth (40×40 lateral
> patch of `vesicles`, peak-normalized, Gaussian noise, Adam 200 it from a ridge start,
> 2026-09-22, `benchmarks/studies/lambda_interpretability.py`); the benchmark repeats them across truths.

---

## 1. The problem: λ has no scale

The shared objective (`core/objective.py`) is a blend:

$$ L(f) = (1-\lambda)\, D(Hf, g) + \lambda\, R(f), \qquad \lambda \in [0, 1]. $$

$D$ is a *scaled* per-pixel negative log-likelihood: it is $\approx \tfrac12$ at the true image
and $0$ at a perfect fit, so $D = O(1)$ whatever the problem, noise or size. $R$ has **no such
scale**: it is a mean of gradients / curvatures of $f$, and its magnitude depends on the
regularizer, on the image's intensity scale, and on how rough $f$ is. On the patch below,
$R_{\text{TV}}(f_0) = 9.8\cdot10^{-3}$ and $R_{\text{Tikhonov}}(f_0) = 1.3\cdot10^{-3}$ — two and three
orders of magnitude smaller than $D$.

So in $(1-\lambda)D + \lambda R$, the prior contributes nothing until $\lambda$ is within a
hair of 1: the weight ratio $\lambda/(1-\lambda)$ has to reach $D/R \sim 10^2$–$10^3$ before $R$
is even felt. **λ is a dead knob over almost all of its range**, and where it finally bites
depends on the regularizer — the opposite of a parameter that transfers.

---

## 2. What λ actually does — measured

`regularization_share` (`benchmarks/metrics.py`) quantifies "how regularized" a result is:

$$ r(\lambda) = 1 - \frac{R(f_\lambda)}{R(f_0)}, \qquad f_0 = \text{the } \lambda=0 \text{ reconstruction}, $$

$0$ at $\lambda=0$, $\to 1$ as the prior wins. Two formulations, swept over $\lambda$:

- **standard** — $R$ used as written;
- **calibrated** — $R$ rescaled by $1/R(f_0)$, i.e. $L = (1-\lambda)D + \lambda\,R/R(f_0)$, so the
  two terms are comparable at $f_0$.

| | standard $r(\lambda)$ | | calibrated $r(\lambda)$ | |
|---|---|---|---|---|
| $\lambda$ | **TV** | **Tikhonov** | **TV** | **Tikhonov** |
| 0.10 | 0.004 | 0.004 | 0.146 | 0.490 |
| 0.25 | 0.009 | 0.012 | 0.285 | 0.669 |
| 0.50 | 0.022 | 0.033 | 0.519 | 0.797 |
| 0.75 | 0.051 | 0.078 | 0.734 | 0.877 |
| 0.90 | 0.123 | 0.163 | 0.814 | 0.939 |
| 0.99 | 0.512 | 0.516 | 0.807 | 0.995 |

Read across:

- **Standard: $r \approx 0$ until $\lambda \gtrsim 0.9$.** Every useful setting is crammed into
  $[0.9, 1.0]$ and the best NMSE for TV sits at $\lambda = 0.90$ (0.259) — a value no user would
  reach for, and not the same value for another prior or truth. λ is not interpretable and
  does not transfer.
- **Calibrated: $\lambda$ reads as the regularization share.** For TV, $r(\lambda) \approx \lambda$
  (nearly the diagonal), and the best NMSE (0.255) lands at $\lambda = 0.10$, $r \approx 0.15$ — a
  human-scale knob with a graded, monotone response. Tikhonov saturates faster ($r = 0.49$
  already at $\lambda = 0.10$) because its $R(f_0)$ is smaller; the mapping is not identical
  across priors, but it is monotone and lands the action in $[0, 1]$ instead of $[0.9, 1]$.

The calibrated form also honestly shows a *wrong* prior: calibrated Tikhonov's NMSE only ever
rises with $\lambda$ (0.38 → 0.89) — L2-on-the-gradient over-smooths these vesicles, and the
graded axis makes that visible instead of hiding it near $\lambda = 1$.

---

## 3. The reference point, and why $R(f_0)$ is the universal one

Calibration needs a reference magnitude for $R$. The pre-study (2026-09-21) tried three:

| reference | interpretable λ? | universal? |
|---|---|---|
| the start $f_{\text{init}}$ | over-strong (the start is badly scaled) | yes |
| a ridge estimate $R(\text{ridge}(s_2))$ | yes | **no** — $s_2$ needs the operator's spectrum, absent for a general problem |
| **the unregularized reconstruction $R(f_0)$** | **yes** | **yes** |

$R(f_0)$ wins on universality: $f_0$ is just the $\lambda=0$ run of the *same solver with the
same settings* — it exists for every inverse problem, every regularizer and every noise model,
and needs no operator spectrum. It is also exactly the denominator of the reported $r(\lambda)$
(§2), so calibrating the objective by $R(f_0)$ makes **the knob the user turns and the share
the benchmark reports the same axis**. Its one cost: it is data-dependent (one extra
unregularized solve per configuration) and it changes what a stored $\lambda$ means.

Scale normalizations of $g$ cancel in the calibrated ratio, so a calibrated $\lambda$ is
insensitive to the intensity normalization — a further point in its favour for transfer across
problems.

---

## 4. D1 — the decision

Whether to change the shared formulation is the project owner's call (the rule: no algorithm
/ formulation change without permission; and `core/objective.py` warns that "fixing" the
weighting silently changes every stored config). The options:

- **A — keep the standard formulation, report $r(\lambda)$ only.** The objective stays the exact
  Bayesian posterior $(1-\lambda)D + \lambda R$; nothing stored changes. Interpretability is
  provided *outside* the objective: the GUI and benchmark show $r(\lambda)$, and the benchmark
  sweeps λ on a log grid near 1 (where the action is). Honest and stable, but λ itself stays
  an expert knob living in $[0.9, 1]$.
- **B — calibrate $R$ by $R(f_0)$ (recommended).** $L = (1-\lambda)D + \lambda\,R/R(f_0)$. λ becomes
  the interpretable, transferable share the project set out to give the user, matching the
  reported $r$. Costs: one extra unregularized solve per run, and **every stored $\lambda$
  changes meaning** (acceptable for v2, a rewrite; must be stated in the migration note).
- **C — calibrate by an operator-derived reference** ($s_2$, ridge). Interpretable but **not**
  transferable across inverse problems, which is a stated v2 goal → not recommended.

**Recommendation: B.** It is the only option that delivers the headline v2 promise — "λ as a
percentage, comparable across problems and priors" — and it costs one cheap extra solve. I
will not implement it without your go: it touches the shared objective and every saved λ.

---

## 5. Either way: what the benchmark measures

Independent of D1, the λ study in the benchmark reports, per (truth, noise, regularizer):
`regularization_share` $r(\lambda)$, NMSE, `chi2_ratio` (over- vs under-fit), and the realism
verdict — so the λ→reconstruction relation is shown quantitatively ("raise λ, get this share,
this fit, this error"), which is the deliverable regardless of the formulation chosen.

## 6. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-L1 | standard: $r(\lambda) \approx 0$ until $\lambda \gtrsim 0.9$; best λ crammed in $[0.9,1]$ and not transferable | λ sweep (log grid near 1) per truth/prior |
| H-L2 | calibrated by $R(f_0)$: $r(\lambda)$ graded and monotone, best λ in a human range, and the best λ transfers across truths, noise levels and priors better than the standard one | λ sweep both formulations, both problems |
| H-L3 | $R(f_0)$ is a stabler reference than $s_2$/ridge across inverse problems (MA-TIRF ↔ deconvolution) | calibrate both ways, compare transfer |
| H-L4 | at the best λ, `chi2_ratio` $\approx 1$ (right fit); over-regularizing pushes it $\gg 1$, under-regularizing $\ll 1$ | chi2_ratio vs λ |
