# PnP (Plug-and-Play HQS) — theory of the parameters, before any benchmark

> Status: **theoretical note** (Day 7, step 2). MA-TIRF spectrum as in `adam_ppxa.md` §1.1
> ($s_1 = 38.9$, $s_2 = 6.02$, $s_3 = 0.562$). PnP is benchmarked on MA-TIRF only (the
> deconvolution benchmark covers Adam, PPXA and MCMC). Nothing was modified; §5 lists
> proposals. The denoiser analysis of §3 also applies to ADMM-PnP and MCMC.

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

## 5. Issues and proposals (not applied)

1. **Scale-free denoising** — apply denoisers to $f$ normalized by its peak,
   $p \cdot D(255 f/p, \sigma)/255$ with $p = \max f$, so that $\sigma$ means "noise level for an
   image whose peak is 255" on every problem. This makes $\sigma$ transferable between
   inverse problems (§3); it changes results for Bilateral, Wiener and DCT on MA-TIRF, not
   on deconvolution. The same change would apply to ADMM-PnP.
2. **Noise-aware final level** — use the measurement's noise level (from the noise model,
   on the 0–255 scale) as $\sigma_{final}$ instead of the hard-coded 1, as in DPIR.
3. **Estimated `delta`** — let the pipeline estimate `delta` for denoisers too (as it does
   for regularizations) when the user leaves it empty.
4. **A proper start** — offer `init`, or start from the ridge estimate, so the null-space
   components of the first iterates have the right scale and shape.

Evidence for 1 is the measurement of §3; 2–4 follow from §1.1–1.2 and are to be confirmed by
the hypotheses below. Should any be wanted, the pattern used for ADMMv2 / MCMCv2 (a separate
PnPv2) keeps the current solver intact for comparison.

---

## 6. Hypotheses the benchmark must test

| # | Hypothesis | Test |
|---|---|---|
| H-P1 | Gaussian and Bilateral (3D) beat TV Bregman (slice by slice) on depth structure | denoiser comparison, depth error |
| H-P2 | Gaussian does nothing for $\sigma \le 7.5$; useful range 10–50 | $\sigma$ sweep, Gaussian |
| H-P3 | Wiener / DCT / Bilateral: best $\sigma$ on MA-TIRF far below their best on deconvolution | $\sigma$ sweep on both problems (denoiser-level test) |
| H-P4 | $\lambda_{kz}$ best in $[10^{-3}, 0.3]$ on MA-TIRF; below: noisy, above: data ignored | $\lambda_{kz}$ sweep |
| H-P5 | results stable for `iter` 8–30; 5 is coarse | `iter` sweep |
| H-P6 | the start matters (adjoint vs ridge) because the null space keeps $z_0$ | start comparison (needs proposal 4 to test inside PnP; otherwise pre-scaling $f_0$ externally) |
| H-P7 | transferability between problems holds for Gaussian / TV Bregman $\sigma$, not for the others | same $\sigma$ on MA-TIRF and deconvolution |
