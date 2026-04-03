# core/reconstruction_metrics.py

import numpy as np

METRICS = {}

def optimal_scale(f, f_true, delta=1., eps=1e-12):
    """
    Calcule alpha* qui minimise ||f_true - alpha f||^2
    en tenant compte de l'anisotropie (delta coefficent d'anisotropie en z).
    """
    # poids volumique (z plus fin → plus de poids)
    # voxel volume ~ dx * dy * dz → proportionnel à delta
    weights = np.ones_like(f)
    weights *= delta  # pondération simple en z
    num = (weights * f * f_true).sum()
    den = (weights * f * f).sum() + eps
    return num / den

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
    f_aligned, _ = align_scale(f, f_true, delta)
    return ((f_aligned - f_true) ** 2).mean().item()

@register_metric("MAE")
def mae_aligned(f, f_true, delta=1.):
    f_aligned, _ = align_scale(f, f_true, delta)
    return (np.abs(f_aligned - f_true)).mean().item()

@register_metric("Correlation")
def correlation(f, f_true, delta=None, eps=1e-12):
    f_flat = f.reshape(-1)
    f_true_flat = f_true.reshape(-1)
    f_mean = f_flat.mean()
    t_mean = f_true_flat.mean()
    num = ((f_flat - f_mean) * (f_true_flat - t_mean)).sum()
    den = (
        np.sqrt(((f_flat - f_mean) ** 2).sum()) *
        np.sqrt(((f_true_flat - t_mean) ** 2).sum()) + eps
    )
    return (num / den).item()

@register_metric("SSIM")
def ssim_3d_aligned(f, f_true, delta=1.):
    f_aligned, _ = align_scale(f, f_true, delta)
    scores = []
    for z in range(f.shape[0]):
        s = ssim(
            f_true[z],
            f_aligned[z],
            data_range=f_true.max() - f_true.min()
        )
        scores.append(s)
    return float(np.mean(scores))

def ssim(**kwargs):
    return 0



def compute_all_metrics(f, f_true, delta=1.):
    results = {}
    f = f.detach().cpu().numpy()
    f_true = f_true.detach().cpu().numpy()
    for name, func in METRICS.items():
        try:
            results[name] = func(f, f_true, delta=delta)
        except Exception as e:
            results[name] = f"Error: {e}"
    return results