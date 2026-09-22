# MCMC — theory of the parameters, and the study that produced MCMCv2

> Status: **theoretical note + a measured study** (Day 7, step 2). MCMC is the one solver
> whose two proposed fixes rest on a study already run (2026-09-21, "make MCMC work"), not
> only on predicted ranges — the results below marked *(measured)* come from it. MA-TIRF
> spectrum as in `adam_ppxa.md` §1.1 ($s_1 = 38.9$, $s_2 = 6.02$, $s_3 = 0.562$,
> $s_4 = 0.033$; the $f$ that explains $g$ peaks at $0.022$–$0.025$). The denoisers and their
> MA-TIRF caveats are analysed once in `pnp.md` §2 and not repeated here. MCMC is the only
> sampler in the benchmark, and the only non-Adam/PPXA method run on **deconvolution** too.
> MCMC itself is unchanged; its fixes exist as the separate solver MCMCv2 (§4).

---

## 1. What MCMC does

Every other solver here returns a **MAP** estimate — the single most probable $f$. MCMC is
the odd one out: it samples the posterior $\pi(f) \propto \exp\!\big(-D(Hf, g)/\beta\big)$ by
Metropolis-Hastings and averages the samples, giving the posterior **mean** (MMSE):

$$ f_{\text{MMSE}} \;\approx\; \frac1T \sum_t f_t . $$

A MAP estimate can sit on a sharp, unrepresentative peak of the posterior; the MMSE average
is smoother, and it comes with an **acceptance rate** that tells you whether the chain
actually explored or stalled. One step of the chain, from the current state $f$:

| step | code | meaning |
|---|---|---|
| data | $z = f + (H^TH + \lambda_{rr} I)^{-1} H^T(g - Hf)$ | pull toward data consistency (a ridge step on the residual) |
| explore | $z \leftarrow z + \sigma\,\varepsilon$, $\varepsilon \sim \mathcal N(0, I)$ | perturb |
| prior | $z \leftarrow D_\sigma(\max(z, 0))$ | positivity, then the denoiser |
| accept | keep $z$ with prob. $\min\!\big(1, e^{(D(f) - D(z))/\beta}\big)$ | Metropolis on the data fidelity |

It uses no explicit `reg`/`lambda_reg`; the prior is entirely the denoiser (§1.3). Like the
other solvers it became problem-agnostic through `ForwardOperator.ridge_inverse`, which is
the data step for every operator (MA-TIRF inverts a dense matrix, deconvolution filters in
Fourier — one line here).

### 1.1 Acceptance is a switch, not a dial — the reason MCMC was fragile

A worse candidate ($D(z) > D(f)$) is accepted with probability $e^{-\Delta D/\beta}$,
$\Delta D = D(z) - D(f) > 0$. So the only thing that matters is $\beta$ **against the typical
$\Delta D$ a proposal causes**:

- $\beta \ll \Delta D$: nothing is accepted; the chain never leaves its start.
- $\beta \gg \Delta D$: almost everything is accepted (90–99 %); the result no longer depends
  on $\beta$.

The useful window is narrow, and — crucially — $\Delta D$ is **not a constant**. $D$ is the
*scaled* likelihood (`solvers/fidelities/base.py`): with Gaussian noise of variance $b$,
$D = \tfrac1{2bN}\lVert Hf-g\rVert^2$. A perturbation of size $\sigma$ moves $Hf$ by $\sim s\,\sigma$,
so

$$ \Delta D \;\sim\; \frac{s^2\,\sigma^2}{b\,N} \;\propto\; \sigma^2 \quad\text{and}\quad \propto \frac1b . $$

A $\beta$ tuned for one $\sigma$ and one noise level is off by orders of magnitude at another.
**This is why v1 "worked with very specific parameters or not at all":** its $\beta$ was a
raw number that had to be hand-matched to $\sigma^2/b$ each time. MCMCv2's calibration (§4)
removes exactly this coupling.

### 1.2 The data step and $\lambda_{rr}$

$(H^TH + \lambda_{rr} I)^{-1}$ corrects the directions with $s_i^2 \gg \lambda_{rr}$ toward the
data and leaves the rest — the same ridge mechanism as ADMM's $\mu$ and the warm-start
weight (`adam_ppxa.md` §5). On MA-TIRF only 2–3 depth directions per column are determined,
so $\lambda_{rr}$ decides how many the chain trusts each step: **$\lambda_{rr} \in [s_2, s_1]$
keeps the determined ones and cuts the noise-dominated rest**; below $s_3$ it amplifies noise
along the poorly determined directions, and far above $s_1$ (v1's $10^4$) it barely pulls at
all, leaving the chain to wander on the denoiser alone.

### 1.3 The denoiser is the prior

There is no $R(f)$: the denoiser removes the noise just injected, at the same level $\sigma$,
and what it keeps is what it considers "image". **Without a denoiser the chain has no prior
and cannot beat its own starting point** — it only adds noise the data step then fights. So
`denoiser = None` is a diagnostic, not a working configuration.

### 1.4 The mean, after a burn-in

The chain needs to forget its start before its states represent the posterior. Standard MCMC
discards an initial **burn-in** and averages the rest; the mean of that stationary part beats
both the last state (one noisy sample) and the mean including burn-in (biased toward the
start). *(measured: the average clearly beats the last state.)*

---

## 2. The parameters

### `denoiser` — the prior *(critical)*

Same catalogue and MA-TIRF caveats as PnP (`pnp.md` §2). *(measured:* TV Bregman is the most
robust across noise levels; Wiener is excellent at low noise and degrades at high noise; DCT
breaks the reconstruction and is rejected by the realism check.*)* `None` = no prior (§1.3).

### `sigma` — exploration **and** denoising strength *(critical)*

One number does two jobs: the std of the perturbation and the denoiser's level. In **v1** it
is on $f$'s own scale (default $0.05$) — but $f$ peaks at $\approx 0.023$ on MA-TIRF, so the
default perturbation is *twice the signal*, and the same $0.05$ means something entirely
different on a photograph (peak $\sim 1$). That is the denoiser-scale problem of `pnp.md` §3,
here on the proposal as well. MCMCv2 makes it a **fraction of the peak** (§4).

- $\sigma \to 0$: no exploration, no denoising — the chain stays at its start.
- $\sigma$ large: over-smoothing; fine structure erased, and $\Delta D$ (∝ $\sigma^2$) explodes,
  collapsing the acceptance (§1.1).

*(measured:* $\sigma \approx 0.02$–$0.05$ of the peak is good; its optimum grows with the noise.*)*

### `beta` $\beta$ — temperature *(critical in v1, by §1.1)*

The exact posterior of the Bayesian model is $\beta = 1/N_g$ ($D$ is a per-pixel negative
log-likelihood). But because acceptance is a switch (§1.1), the *practical* value is whatever
makes $\Delta D/\beta$ order 1 — which depends on $\sigma$ and $b$. v1 exposes $\beta$ as a raw
number (default $0.01$); MCMCv2 calibrates it instead (§4).

### `lambda_rr` — ridge weight of the data step *(mostly fixed by a rule)*

§1.2. **Predicted / measured useful window $\lambda_{rr} \in [s_2, s_1] \approx [6, 39]$ on
MA-TIRF.** v1's default $10^4 \gg s_1$ barely pulls — the chain then leans almost entirely on
the denoiser. MCMCv2 defaults to the automatic $s_1$ (top of the window).

### `max_iter` $T$ *(comfort, within limits)* and `K`

*(measured:* the chain converges in $\approx 150$ iterations.*)* `K` only sets the logging /
preview cadence. v1 has no burn-in; it averages the **accepted** samples over the whole run.

### `init` — the start *(not critical)*

*(measured:* adjoint, ridge($10^4$) and ridge($s_2$) starts all reach the same mean — the
burn-in erases the start.*)* Offered for completeness.

### `delta` — anisotropy *(fixed by the physics)*

As for PnP: the axial-to-lateral kernel ratio for Gaussian / Bilateral; defaults to 1 (v1
does not estimate it).

---

## 3. Summary

| parameter | role | critical? | range (MA-TIRF) | toward small | toward large |
|---|---|---|---|---|---|
| `denoiser` | the prior (there is no $R$) | **yes** | TV Bregman robust; Wiener low-noise; `None` = no prior | — | — |
| `sigma` | exploration × denoising | **yes** | 0.02–0.05 of the peak | chain frozen at start | structure erased, acceptance collapses |
| `beta` | temperature (a switch, §1.1) | **yes (v1)** | matched to $\Delta D \propto \sigma^2/b$ | nothing accepted | everything accepted (~99 %) |
| `lambda_rr` | data-step ridge | by rule | $[s_2, s_1] \approx [6, 39]$ | noise along $s_3, s_4$ | barely pulls (v1's $10^4$) |
| `max_iter` $T$ | budget | comfort | $\gtrsim 150$ | not converged | plateau |
| `init` | start | no | any | — | — |

For a user, MCMC comes down to **which denoiser and how much to explore ($\sigma$)** — provided
$\beta$ is matched to the resulting $\Delta D$. Getting that last match right by hand is what
made v1 fragile, and what MCMCv2 automates.

---

## 4. MCMCv2 — the proposals, as a separate solver

At the project owner's request the fixes exist as the solver "MCMCv2" (`solvers/mcmc_v2.py`);
MCMC is unchanged. Each fix was **measured before being made**. The benchmark compares them
and keeps one if the other is clearly worse.

| | MCMC | MCMCv2 |
|---|---|---|
| $\beta$ | raw number (default $0.01$), matched by hand to $\sigma^2/b$ | **calibrated**: $\beta = \tau \times \mathrm{median}\lvert\Delta D\rvert$ over the first 5 proposals; $\tau$ (`temperature`) is relative |
| $\sigma$ | on $f$'s own scale (default $0.05$) | **fraction of the peak** (default $0.03$), scale-free across problems |
| $\lambda_{rr}$ | default $10^4 \gg s_1$ (barely pulls) | empty = automatic $s_1$ |
| estimate | mean of the **accepted** samples, no burn-in | mean of **all** states after a 25 % burn-in (standard MCMC) |

The point of the calibration: since $\Delta D \propto \sigma^2/b$ (§1.1), tying $\beta$ to the
measured $\Delta D$ makes a single relative $\tau$ work at every $\sigma$, noise level and
inverse problem — the switch is set automatically to the middle of its window.

*Evidence (`solvers/_tests.py`, `test_mcmc_is_robust_to_its_temperature`)*: with the
calibration on, temperatures $\tau \in \{0.3, 1.0, 10\}$ all keep the chain moving (acceptance
$> 50\%$, in practice 90–100 %) — whereas a raw $\beta$ spanning $30\times$ would swing from a
frozen to a saturated chain. The automatic $\lambda_{rr}$ equals the largest singular value to
$10^{-3}$, and the meaningless `proposal_method` switch is gone from both (it changed nothing:
both orders agreed to three decimals — see `solvers/mcmc.py`).

## 5. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-M1 | acceptance is a switch: best $\beta \propto \sigma^2/b$; MCMCv2's calibrated $\tau$ tracks it across $\sigma$ and noise, a fixed $\beta$ does not | $\beta$ (MCMC) and $\tau$ (MCMCv2) sweeps × noise levels, acceptance + NMSE |
| H-M2 | TV Bregman is the most robust prior across noise; Wiener best at low noise; `None` never beats its start; DCT rejected | denoiser comparison per noise level, realism check |
| H-M3 | best $\lambda_{rr} \in [s_2, s_1]$; below $s_3$ noisy, $\gg s_1$ leans on the denoiser | $\lambda_{rr}$ sweep, depth error |
| H-M4 | the chain plateaus by $\approx 150$ iterations; the post-burn-in mean beats the last state and the accepted-only mean | iterate history, three estimators compared |
| H-M5 | MCMCv2's peak-relative $\sigma$ transfers between MA-TIRF and deconvolution; MCMC's absolute $\sigma$ does not | same $\sigma$ on both problems, NMSE |
| H-M6 | MCMC (MMSE) vs Adam/PPXA (MAP): the mean is smoother and more robust to noise, at a higher cost | same truths/noise, NMSE + realism + time |
| H-M7 | MCMCv2 ≥ MCMC on both problems, and is tunable with one relative $\tau$ where MCMC needs a matched $\beta$ | MCMC vs MCMCv2 at each one's best settings, sensitivity to the temperature knob |
