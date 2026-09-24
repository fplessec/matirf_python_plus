"""
The benchmark's metrics — few, each answering one question.

    nmse               how wrong overall?        ||alpha f - f_true||^2 / ||f_true||^2, alpha
                                                 the best scale (MA-TIRF fixes f only up to a
                                                 positive factor); 0 perfect, 1 = no better
                                                 than zero
    psnr               how wrong overall (dB)?   10 log10(peak^2 / MSE) after the same best-
                                                 scale alignment, peak = the truth's max. The
                                                 SAME information as nmse (monotone in it, same
                                                 ranking) but in the dB currency image
                                                 processing reads at a glance — a companion,
                                                 not an independent axis
    ssim               structurally similar?     2D structural similarity (skimage), on the
                                                 aligned f. DECONVOLUTION ONLY: reliable in 2D,
                                                 unreliable on the anisotropic MA-TIRF volume
    depth_error_nm     at the right depth?       mean |z(f) - z(f_true)| over the pixels that
                                                 hold signal, z = depth of the brightest voxel
                                                 of the column — THE question of MA-TIRF
                                                 ("how far from the glass is it?")
    stack_recovery     super-resolved in z?      of the pixels where the truth holds two
                                                 structures >= 30 nm apart in depth, the
                                                 fraction where f shows two peaks too
    chi2_ratio         fitting the data or the   mean residual^2 / noise variance: ~1 is the
                       noise?                    right fit, << 1 over-fits the noise, >> 1
                                                 over-regularizes; needs no truth, so it also
                                                 reads on real data
    regularization_share  how regularized?       r = 1 - R(f_lambda) / R(f_0), f_0 the
                                                 unregularized reconstruction: 0 at lambda = 0,
                                                 -> 1 when the prior wins. What "lambda_reg =
                                                 0.1 means 10 % regularized" is tested against

Reported per run by the runner (practicality, not quality): runtime, peak memory, status
(ok / rejected / failed / timeout) — a solver that OOMs on a real stack is disqualified by that
alone, whatever its nmse would have been.

Dropped on purpose: MSE / MAE (redundant with nmse, and not scale-invariant), FSC (H acts only
in depth: lateral resolution is not the subject), 3D SSIM (unreliable on an anisotropic volume —
ssim is kept for the 2D deconvolution only), Sinkhorn (costly; depth_error and the realism
check's depth shift say the same).
"""

import math

import torch

## a column holds signal above this fraction of the volume's brightest column
SIGNAL = 0.1
## two peaks count as separate structures at this many planes apart (5 x 6 nm = 30 nm)
SEPARATION_PLANES = 5
## a local maximum counts as a peak above this fraction of its column's maximum
PEAK = 0.2


def nmse(f: torch.Tensor, truth: torch.Tensor) -> float:
    f, truth = f.detach().double(), truth.detach().double()
    alpha = float((f * truth).sum() / (f * f).sum().clamp(min=1e-300))
    return float(((alpha * f - truth) ** 2).sum() / (truth ** 2).sum())


def psnr(f: torch.Tensor, truth: torch.Tensor) -> float:
    """Peak SNR in dB, after the same best-scale alignment as nmse (reference peak = truth.max).

    Monotone in nmse, so it ranks solvers identically — reported because dB is the currency of
    image processing, not because it adds discriminative information."""
    f, truth = f.detach().double(), truth.detach().double()
    alpha = float((f * truth).sum() / (f * f).sum().clamp(min=1e-300))
    mse = float(((alpha * f - truth) ** 2).mean())
    peak = float(truth.max())
    if peak <= 0:
        return float("nan")
    return float("inf") if mse <= 0 else float(10.0 * math.log10(peak ** 2 / mse))


def ssim(f: torch.Tensor, truth: torch.Tensor) -> float:
    """2D structural similarity (skimage), after best-scale alignment. Deconvolution only.

    Not used on MA-TIRF: SSIM's isotropic window is unreliable on the anisotropic z x xy volume."""
    from skimage.metrics import structural_similarity

    f, truth = f.detach().double(), truth.detach().double()
    alpha = float((f * truth).sum() / (f * f).sum().clamp(min=1e-300))
    aligned = (alpha * f).cpu().numpy()
    reference = truth.cpu().numpy()
    data_range = float(reference.max() - reference.min()) or 1.0
    return float(structural_similarity(reference, aligned, data_range=data_range))


def _signal_columns(truth: torch.Tensor) -> torch.Tensor:
    lateral = truth.max(dim=0).values
    return lateral > SIGNAL * lateral.max()


def depth_error_nm(f: torch.Tensor, truth: torch.Tensor, dz_nm: float) -> float:
    columns = _signal_columns(truth)
    z_f, z_t = f.argmax(dim=0)[columns].double(), truth.argmax(dim=0)[columns].double()
    return float((z_f - z_t).abs().mean() * dz_nm) if columns.any() else float("nan")


def _stacked(volume: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """For each selected column: does it hold two peaks >= SEPARATION_PLANES apart?"""
    col = volume[:, columns]
    inner = col[1:-1]
    peaks = (inner > col[:-2]) & (inner >= col[2:]) & (inner > PEAK * col.max(dim=0).values)
    index = torch.arange(1, col.shape[0] - 1, device=col.device)[:, None].expand_as(peaks)
    first = torch.where(peaks, index, col.shape[0]).min(dim=0).values
    last = torch.where(peaks, index, -1).max(dim=0).values
    return (last - first) >= SEPARATION_PLANES


def stack_recovery(f: torch.Tensor, truth: torch.Tensor) -> float:
    """Fraction of the truth's stacked columns that f also resolves; nan if there are none."""
    columns = _signal_columns(truth)
    stacked_truth = _stacked(truth, columns)
    if not stacked_truth.any():
        return float("nan")
    return float(_stacked(f, columns)[stacked_truth].double().mean())


def chi2_ratio(f: torch.Tensor, operator, g: torch.Tensor, a: float, b: float) -> float:
    """mean (Hf - g)^2 / Var, Var = a Hf + b — the noise model's own variance (a, b)."""
    Hf = operator.apply(f.detach())
    variance = (a * Hf.clamp(min=0) + b).clamp(min=1e-12)
    ## Hf is only defined up to the scale of f (MA-TIRF): align it to g first
    alpha = float((Hf * g).sum() / (Hf * Hf).sum().clamp(min=1e-300))
    return float(((alpha * Hf - g) ** 2 / variance).mean())


def regularization_share(R_lambda: float, R_zero: float) -> float:
    """1 - R(f_lambda) / R(f_0): how much of the unregularized irregularity was removed."""
    return float(1.0 - R_lambda / R_zero) if R_zero > 0 else float("nan")
