import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np

from settings import device


METRICS = {}


## computes alpha* that minimizes ||f_true - alpha * f||^2:
def optimal_scale(f, f_true, delta=1., eps=1e-12):
    f = torch.as_tensor(f, device=device, dtype=torch.float32)
    f_true = torch.as_tensor(f_true, device=device, dtype=torch.float32)
    num = (f * f_true).sum()
    den = (f * f).sum() + eps
    return (num / den).item()


def align_scale(f, f_true, delta=1.):
    alpha = optimal_scale(f, f_true, delta=delta)
    return alpha * f, alpha


def register_metric(name):
    def decorator(func):
        METRICS[name] = func
        return func
    return decorator




@register_metric("Scale_alpha")
def scale_alpha(f, f_true, delta=1.):
    return optimal_scale(f, f_true, delta)


@register_metric("MSE")
def mse_aligned(f, f_true, delta=1.):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    f_aligned, _ = align_scale(f, f_true, delta)
    return torch.mean((f_aligned - f_true) ** 2).item()


@register_metric("MAE")
def mae_aligned(f, f_true, delta=1.):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    f_aligned, _ = align_scale(f, f_true, delta)
    return torch.mean(torch.abs(f_aligned - f_true)).item()


@register_metric("NMSE")
def normalised_mse(f, f_true, delta=None):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    alpha = optimal_scale(f, f_true)
    num = torch.sum((alpha * f - f_true) ** 2)
    den = torch.sum(f_true ** 2) + 1e-12
    return (num / den).item()


@register_metric("angular distance")
def angular_distance(f, f_true, delta=None, eps=1e-12):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    f_flat = f.reshape(-1)
    t_flat = f_true.reshape(-1)
    dot = torch.sum(f_flat * t_flat)
    norm_f = torch.norm(f_flat)
    norm_t = torch.norm(t_flat)
    cos = dot / (norm_f * norm_t + eps)
    cos = torch.clamp(cos, -1, 1)
    return torch.acos(cos).item()


@register_metric("cosine similarity")
def cosine_similarity(f, f_true, delta=None, eps=1e-12):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    f_flat = f.reshape(-1)
    t_flat = f_true.reshape(-1)
    dot = torch.sum(f_flat * t_flat)
    norm_f = torch.norm(f_flat)
    norm_t = torch.norm(t_flat)
    return (dot / (norm_f * norm_t + eps)).item()


@register_metric("Correlation")  # Pearson
def correlation(f, f_true, delta=None, eps=1e-12):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    f_flat = f.reshape(-1)
    t_flat = f_true.reshape(-1)
    f_mean = torch.mean(f_flat)
    t_mean = torch.mean(t_flat)
    num = torch.sum((f_flat - f_mean) * (t_flat - t_mean))
    den = torch.sqrt(torch.sum((f_flat - f_mean) ** 2)) * \
          torch.sqrt(torch.sum((t_flat - t_mean) ** 2)) + eps
    return (num / den).item()


@register_metric("PNSR")
def psnr_invariant(f, f_true, delta=None):
    f = torch.as_tensor(f, device=device)
    f_true = torch.as_tensor(f_true, device=device)
    alpha = optimal_scale(f, f_true)
    mse = torch.mean((alpha * f - f_true) ** 2)
    max_val = torch.max(f_true)
    return (10 * torch.log10((max_val ** 2) / (mse + 1e-12))).item()


@register_metric("SSIM")
def ssim_3d_aligned(f, f_true, delta=1.):
    f_aligned, _ = align_scale(f, f_true, delta)
    # SSIM is numpy-based
    f_np = f_aligned.detach().cpu().numpy()
    t_np = torch.as_tensor(f_true).detach().cpu().numpy()
    scores = []
    for z in range(f_np.shape[0]):
        s = ssim(
            t_np[z],
            f_np[z],
            data_range=t_np.max() - t_np.min()
        )
        scores.append(s)
    return float(np.mean(scores))


@register_metric("Sinkhorn Wasserstein")
def sinkhorn_wasserstein_3d(
    f,
    f_true,
    delta=1.0,
    reg=1e-2,
    max_iter=100,
    tol=1e-6,
    eps=1e-12,
    max_points=5000,
):
    # TODO: review (optimal transport)
    f = torch.as_tensor(f, device=device).float()
    f_true = torch.as_tensor(f_true, device=device).float()
    Z, Y, X = f_true.shape
    f = f.reshape(-1)
    t = f_true.reshape(-1)
    f_sum = f.sum()
    t_sum = t.sum()
    if f_sum < eps or t_sum < eps:
        return float("nan")
    mu = f / f_sum
    nu = t / t_sum
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, device=device),
        torch.arange(Y, device=device),
        torch.arange(X, device=device),
        indexing='ij'
    )
    coords = torch.stack([
        zz * delta,
        yy,
        xx
    ], dim=-1).reshape(-1, 3).float()
    idx_f = torch.where(mu > 0)[0]
    idx_t = torch.where(nu > 0)[0]
    if len(idx_f) > max_points:
        idx_f = idx_f[torch.randperm(len(idx_f))[:max_points]]
    if len(idx_t) > max_points:
        idx_t = idx_t[torch.randperm(len(idx_t))[:max_points]]
    coords_f = coords[idx_f]
    coords_t = coords[idx_t]
    mu = mu[idx_f]
    nu = nu[idx_t]
    mu = mu / (mu.sum() + eps)
    nu = nu / (nu.sum() + eps)
    C = torch.cdist(coords_f, coords_t) ** 2
    K = torch.exp(-C / reg)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(max_iter):
        u_prev = u
        u = mu / (K @ v + eps)
        v = nu / (K.t() @ u + eps)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break
    gamma = torch.diag(u) @ K @ torch.diag(v)
    W = torch.sum(gamma * C)
    return torch.sqrt(W).item()  # average transport distance in pixel/voxel
    return W.item()


@register_metric("FSC_mean")
# TODO: review (Fourier shell correlation)
def fsc_3d(f, f_true, delta=None, n_shells=50, eps=1e-12):
    f = torch.as_tensor(f, device=device).float()
    f_true = torch.as_tensor(f_true, device=device).float()
    # FFT
    F1 = torch.fft.fftn(f)
    F2 = torch.fft.fftn(f_true)
    # frequencies
    Z, Y, X = f.shape
    zz, yy, xx = torch.meshgrid(
        torch.fft.fftfreq(Z, device=device),
        torch.fft.fftfreq(Y, device=device),
        torch.fft.fftfreq(X, device=device),
        indexing='ij'
    )
    r = torch.sqrt(zz**2 + yy**2 + xx**2)
    r = r.reshape(-1)
    F1 = F1.reshape(-1)
    F2 = F2.reshape(-1)
    r_max = r.max()
    bins = torch.linspace(0, r_max, n_shells+1, device=device)
    fsc_vals = []
    for i in range(n_shells):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.sum() < 10:
            continue
        num = torch.sum(F1[mask] * torch.conj(F2[mask]))
        den = torch.sqrt(
            torch.sum(torch.abs(F1[mask])**2) *
            torch.sum(torch.abs(F2[mask])**2)
        ) + eps
        fsc = torch.real(num / den)
        fsc_vals.append(fsc)
    if len(fsc_vals) == 0:
        return float("nan")
    return torch.mean(torch.stack(fsc_vals)).item()




def compute_all_metrics(f, f_true, delta=1.):
    results = {}
    f = f.detach()
    f_true = f_true.detach()
    for name, func in METRICS.items():
        try:
            results[name] = func(f, f_true, delta=delta)
        except Exception as e:
            results[name] = f"Error: {e}"
    return results