# PnP (Plug-and-Play HQS) — theory of the parameters, before any benchmark

> Status: **theoretical note** (Day 7, step 2). MA-TIRF spectrum as in `adam_ppxa.md` §1.1
> ($s_1 = 38.9$, $s_2 = 6.02$, $s_3 = 0.562$). PnP is benchmarked on MA-TIRF only (the
> deconvolution benchmark covers Adam, PPXA and MCMC). PnP itself is unchanged; its proposed
> fixes exist as the separate solver PNPv2 (§5). The denoiser analysis of §3 also applies
> to ADMM-PnP and MCMC. DCT is considered broken (to be redone): offered by both solvers,
> but not used in the benchmark.

---

## 1. What PnP does

Plug-and-Play (Venkatakrishnan et al., 2013) keeps the structure of a splitting algorithm but
replaces the proximal operator of an explicit prior by **a denoiser**. The prior is then
whatever the denoiser implicitly believes an image looks like.

This solver is **Half-Quadratic Splitting (HQS)** with the annealing schedule of Zhang et al.
(DPIR, 2021). With $z$ the denoised estimate, iteration $k$ is

| step | code | meaning |
|---|---|---|
| data | $f = (H^TH + \alpha_k I)^{-1}(H^Tg + \alpha_k z)$ | fit the data, pulled toward $z$ with weight $\alpha_k$ |
| prior | $z = D_{\sigma_k}(f)$ | denoise at level $\sigma_k$ |
| positivity | $f, z \leftarrow \max(\cdot, 0)$ | if `forced_pos` |

HQS is the splitting of $\min_f \tfrac12\lVert Hf-g\rVert^2 + \Phi(f)$ with $f = z$ enforced by a
quadratic penalty $\tfrac{\alpha}{2}\lVert f - z\rVert^2$; the $z$-step is
$\mathrm{prox}_{\Phi/\alpha}$, which a Gaussian denoiser of level $\sigma$ approximates when
$\sigma^2 \propto 1/\alpha$. The data term is the plain $\tfrac12\lVert Hf-g\rVert^2$ (Gaussian
noise only); PnP uses neither `reg`/`lambda_reg` nor the noise model's level.

### 1.1 The data step, direction by direction

Along singular direction $i$, the data step returns

$$ f_i = \frac{s_i^2}{s_i^2 + \alpha}\, \frac{(H^Tg)_i}{s_i^2} \;+\; \frac{\alpha}{s_i^2 + \alpha}\, z_i . $$

Directions with $s_i^2 \gg \alpha$ follow the data; directions with $s_i^2 \ll \alpha$ **keep $z$ as
it is**. On MA-TIRF, 47 of the 50 depth directions per column are of the second kind at any
reasonable $\alpha$: **the reconstruction's depth structure is essentially the denoiser's**.

### 1.2 The schedule (`kai_zhang = True`)

The code sets $\sigma_{final} = 1$ and, over `iter` steps, logarithmically

$$ \alpha_k:\ \lambda_{kz}\,\frac{1}{\sigma^2} \;\nearrow\; \lambda_{kz}, \qquad
   \sigma_k = \sqrt{\lambda_{kz}/\alpha_k}:\ \sigma \;\searrow\; 1, \qquad
   \alpha_k \sigma_k^2 = \lambda_{kz}\ \text{(constant)}. $$

It starts with a weak data term and strong denoising, and ends with the data trusted at weight
$\lambda_{kz}$ and a light denoising at level 1. In DPIR the final level is **the noise level of
the measurement** ($\sigma_n$ on the 0–255 scale) and $\alpha_k = \lambda\,\sigma_n^2/\sigma_k^2$; here
$\sigma_{final} = 1$ is hard-coded, i.e. PnP always assumes a noise of $1/255 \approx 0.4\,\%$ of the
peak, whatever the data.

---

## 2. The parameters

### `denoiser` — the prior *(critical)*

The single most important choice (§1.1). Available here (NL-Ridge excluded: GPU-bound):

| denoiser | acts through σ as | 3D | effect of the image's intensity scale (§3) |
|---|---|---|---|
| Gaussian | a spatial width $\mathrm{clip}(\sigma/25, 0.3, 2.5)$ px | yes | none |
| TV Bregman | a data weight $\mathrm{clip}(3/\sigma, 0.005, 1)$ | **no** (slice by slice) | none (measured) |
| Bilateral | an intensity tolerance $2\sigma$ (+ a bounded width) | yes | strong |
| Wiener | a gain $\max(\mathrm{var} - \sigma^2, 0)/\mathrm{var}$ | yes | very strong |
| DCT | a threshold $\sigma\sqrt{2\log N}$ | yes | very strong |

Two MA-TIRF-specific consequences:

- **TV Bregman works slice by slice**: it never smooths along z, while z is exactly what the
  denoiser must structure (§1.1). Expected: poor depth structure.
- **Gaussian is a no-op for $\sigma \le 7.5$** (width pinned at 0.3 px): with the default
  $\sigma = 5$ and a schedule going down to 1, the Gaussian denoiser does nothing at all.

### `sigma` — initial denoising level *(critical)*

On the 0–255 scale (`solvers/denoising.py`). The schedule goes from `sigma` down to 1.

- $\sigma \to 0$ (≤ 1): no denoising, no prior → the null space keeps the start $z_0$ (§1.1)
  → the result is the start's depth structure plus least squares.
- $\sigma$ large: early iterates are strongly smoothed; the schedule then relaxes it. Too
  large and structures are erased before the data can restore them (the data only restore
  the 2–3 determined directions).

Its meaning depends on the denoiser and — for three of them — on the image's intensity
scale (§3). **Predicted useful range: $\sigma \in [10, 50]$ for Gaussian (below 7.5 it does
nothing), $[2, 15]$ for TV Bregman; for Bilateral / Wiener / DCT no fixed range carries over
between problems** (§3).

### `lambda_kz` — the final data weight *(critical)*

The last iteration's $\alpha = \lambda_{kz}$; by §1.1 it decides which directions follow the data
at the end: those with $s_i^2 \gg \lambda_{kz}$. It also sets the constant product
$\alpha_k\sigma_k^2$, i.e. the balance data/prior along the whole schedule.

- $\lambda_{kz} \to 0$: only the data at the end; noise enters along $s_3, s_4$ (≈ 0.3, 0.001).
- $\lambda_{kz} \gg s_2^2 = 36$: even the determined directions follow the denoiser.

**Predicted range on MA-TIRF: $\lambda_{kz} \in [s_4^2, s_3^2] \approx [10^{-3}, 0.3]$** (trust
$s_1, s_2$, and $s_3$ partly). The default 0.23 ≈ $s_3^2$ is at the top of it. For a normalized
PSF (deconvolution) the same logic gives a Wiener-like $\lambda_{kz}$ of the order of the
noise-to-signal power.

### `iter` — number of annealing steps *(comfort, within limits)*

The schedule is spread over `iter` steps: more steps = a finer annealing, not a longer
optimization toward a fixed objective (HQS with a schedule is not run to convergence).
DPIR uses 8–40. **Predicted: results stable for `iter` $\in [8, 30]$; 5 (default) is coarse.**

### `kai_zhang` — the schedule on/off *(keep fixed: on)*

Off: $\alpha = \lambda_{kz}$ and $\sigma$ constant — plain HQS, which has no convergence guarantee
with a denoiser and tends to over- or under-smooth depending on $\sigma$. The schedule is what
makes HQS usable. **Keep `True`.**

### `forced_pos` — positivity *(keep fixed: on)*

Images are non-negative; clamping $f$ and $z$ removes the negative oscillations of the data
step. **Keep `True`.**

### `delta` — anisotropy for Gaussian and Bilateral *(fixed by the physics — but see §5)*

Sets the axial width of the kernel relative to the lateral one (e.g. Gaussian:
$s_z = \mathrm{clip}(s_{xy}/\delta, \cdot, 3)$). It should be the **estimated** anisotropy ratio,
like Adam/PPXA's `delta` — but for denoisers it is read as-is with a default of 1, and the
pipeline's automatic estimate is applied only to regularizations (§5).

### Not offered: `init`

PnP starts from `init = adjoint` ($z_0 = H^Tg$), ~850× too large on MA-TIRF and badly shaped
in depth. By §1.1 the null-space components of the first iterates **are** $z_0$'s: the first
denoising steps start from a wrong-scaled, wrong-shaped volume (§5).

---

## 3. The denoiser scale: why σ does not transfer between problems

The project's contract (`solvers/denoising.py`): a denoiser is applied as
$D(255 f, \sigma)/255$, assuming **$f \in [0, 1]$**. True for deconvolution (a photograph), false
for MA-TIRF, where $f$ peaks at ≈ 0.02 (`adam_ppxa.md` §1.2) — so MA-TIRF images reach the
denoisers on a 0–5 scale while $\sigma$ is read on 0–255.

*Measured* — relative change $\lVert D(f) - f\rVert/\lVert f\rVert$ at $\sigma = 5$, same noisy image,
peak 0.022 (MA-TIRF scale) vs peak 1 (deconvolution scale):

| | Gaussian | TV Bregman | Bilateral | Wiener | DCT |
|---|---|---|---|---|---|
| peak 0.022 | 0.01 | 0.20 | 0.22 | **0.84** | **0.99** (image erased) |
| peak 1 | 0.01 | 0.20 | 0.09 | 0.20 | 0.40 |

So for Bilateral, Wiener and DCT the **same $\sigma$ is a different prior on MA-TIRF and on
deconvolution**: parameters tuned on one problem do not carry to the other. It also explains
why DCT was systematically rejected by the realism check on MA-TIRF (pre-study). Gaussian and
TV Bregman are scale-free.

---

## 4. Summary

| parameter | role | critical? | predicted range (MA-TIRF) | toward small | toward large |
|---|---|---|---|---|---|
| `denoiser` | the prior | **yes** | Gaussian (σ ≥ 10) or Bilateral (3D); TV Bregman ignores z | — | — |
| `sigma` | initial denoising level | **yes** | Gaussian 10–50, TV Bregman 2–15; others: not transferable | no prior: the start's structure | structures erased |
| `lambda_kz` | final data weight | **yes** | $10^{-3}$ – 0.3 | noise along $s_3, s_4$ | data ignored |
| `iter` | annealing steps | comfort | 8 – 30 | coarse schedule | — |
| `kai_zhang` | schedule | fixed: on | — | — | — |
| `forced_pos` | positivity | fixed: on | — | — | — |
| `delta` | axial kernel width | fixed: estimated | — | — | — |

For a user, PnP comes down to **three decisions: which denoiser, how strong at the start
($\sigma$), and how much to trust the data at the end ($\lambda_{kz}$)**.

---

## 5. PNPv2 — the proposals, as a separate solver

At the project owner's request the proposals exist as `solvers/pnp_v2.py` ("PNPv2"); PnP is
unchanged. The benchmark compares them.

| | PnP | PNPv2 |
|---|---|---|
| denoiser input | $255 f$ — assumes $f \in [0,1]$ | $255 f / p$, $p$ = peak of the start — **scale-free** |
| final level $\sigma_{final}$ | 1 (hard-coded) | the measurement's noise level relative to its peak, $255\,\mathrm{std}(n)/\max g$, estimated from $g$ (or set by hand) |
| `delta` for Gaussian / Bilateral | 1 by default | the operator's estimate when left empty |
| start | $H^Tg$ (~850× too large on MA-TIRF) | ridge, $\lambda_{rr} = s_2^2$ by default (or `adjoint`) |
| schedule, positivity | switchable | always on |
| default `sigma`, `iter` | 5, 5 | 25, 16 |

**Why $s_2^2$ and not $s_2$.** $\lambda_{rr}$ is compared to the eigenvalues $s_i^2$ of $H^TH$, so
$s_2^2$ (the second eigenvalue) is the choice that scales with the operator's gain — hence
transferable. On MA-TIRF $s_2^2 = 36$ keeps $s_1$ fully and $s_2$ half-way, and lies inside the
$[s_2, s_1] = [6, 39]$ range the MCMC study found good. Operators that do not expose their
spectrum fall back on $s_1^2$.

*Evidence (`solvers/_tests.py`)*: on a 2D blur, multiplying $H$'s gain by 50 (with $\lambda_{kz}$
scaled by $50^2$ accordingly) leaves PNPv2's reconstruction exactly divided by 50 (gap
$3\cdot10^{-6}$); PnP's changes by 49 %.

**Open point, not changed.** $\lambda_{kz}$ is still absolute: it is compared to the
eigenvalues $s_i^2$ (§1.1), so it does not transfer across operators of different gain.
Expressing it relative to $s_1^2$ would make it gain-free, but the best value would still
differ between problems whose spectra decay differently (MA-TIRF: 2–3 useful directions per
column; a blur: a continuous spectrum). To be settled by H-P4 and H-P7.

## 6. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-P1 | Gaussian and Bilateral (3D) beat TV Bregman (slice by slice) on depth structure | denoiser comparison, depth error |
| H-P2 | Gaussian does nothing for $\sigma \le 7.5$; useful range 10–50 | $\sigma$ sweep, Gaussian |
| H-P3 | Wiener / DCT / Bilateral: best $\sigma$ on MA-TIRF far below their best on deconvolution | $\sigma$ sweep on both problems (denoiser-level test) |
| H-P4 | $\lambda_{kz}$ best in $[10^{-3}, 0.3]$ on MA-TIRF; below: noisy, above: data ignored | $\lambda_{kz}$ sweep |
| H-P5 | results stable for `iter` 8–30; 5 is coarse | `iter` sweep |
| H-P6 | the start matters (adjoint vs ridge) because the null space keeps $z_0$ | PNPv2 with `init` = adjoint vs ridge |
| H-P7 | transferability between problems holds for Gaussian / TV Bregman $\sigma$, not for the others | same $\sigma$ on MA-TIRF and deconvolution |
| H-P8 | PNPv2 ≥ PnP on MA-TIRF (scale-free denoisers, proper start), equal on deconvolution for Gaussian / TV Bregman | PnP vs PNPv2, same settings |
