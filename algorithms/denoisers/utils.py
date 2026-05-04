import torch
import torch.nn.functional as F

from settings import device, dtype


def _to_5d(y): return y.unsqueeze(0).unsqueeze(0)
def _from_5d(y): return y.squeeze(0).squeeze(0)

def _is_3d(y): return y.shape[0] > 1

def _gaussian_1d(sigma, truncate=3.0):
    r = int(truncate * sigma + 0.5)
    x = torch.arange(-r, r+1, device=device, dtype=dtype)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    return gauss / gauss.sum()

def _kernel_gaussian(sigma_xy, sigma_z=None, is3d=False):
    gauss_xy = _gaussian_1d(sigma_xy)
    if is3d:
        gauss_z = _gaussian_1d(sigma_z)
        return torch.einsum('i,j,k->ijk', gauss_z, gauss_xy, gauss_xy)
    else:  # 2d
        return torch.outer(gauss_xy, gauss_xy)

def _conv_reflect_5d(y5D, K):
    if K.dim() == 3:
        kZ, kY, kX = K.shape
        pad = (kX//2, kX//2, kY//2, kY//2, kZ//2, kZ//2)
        w = K.view(1,1,kZ,kY,kX).to(device=device, dtype=dtype)
        return F.conv3d(F.pad(y5D, pad, mode='reflect'), w)
    else:
        kY, kX = K.shape
        pad = (kX//2, kX//2, kY//2, kY//2, 0, 0)
        w = K.view(1,1,1,kY,kX).to(device=device, dtype=dtype)
        return F.conv3d(F.pad(y5D, pad, mode='reflect'), w)