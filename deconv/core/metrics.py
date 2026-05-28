"""
Métriques de qualité pour la déconvolution 2D.

On utilise scikit-image pour SSIM et PSNR (standard).
Les images sont supposées être dans [0, 1] (donc data_range=1.0).
"""

import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def _to_numpy(x: torch.Tensor):
    return x.detach().to('cpu').float().numpy()


def compute_psnr(f: torch.Tensor, f_true: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (en dB)."""
    return float(psnr(_to_numpy(f_true), _to_numpy(f), data_range=1.0))


def compute_ssim(f: torch.Tensor, f_true: torch.Tensor) -> float:
    """Structural Similarity Index. Valeur dans [-1, 1], 1 = identiques."""
    return float(ssim(_to_numpy(f_true), _to_numpy(f), data_range=1.0))


def compute_mse(f: torch.Tensor, f_true: torch.Tensor) -> float:
    """Mean Squared Error."""
    return float(((f - f_true) ** 2).mean().item())


def compute_all_metrics(f: torch.Tensor, f_true: torch.Tensor) -> dict:
    """Renvoie un dict avec PSNR, SSIM, MSE."""
    return {
        "PSNR": compute_psnr(f, f_true),
        "SSIM": compute_ssim(f, f_true),
        "MSE": compute_mse(f, f_true),
    }
