import torch
import torch.nn.functional as F

from settings import device,dtype
from .utils import _from_5d, _to_5d, _is_3d, _kernel_gaussian


def _shift_tensor(x5d, dz, dy, dx):
    """Shift 5D tensor (B,C,Z,Y,X) with reflect padding"""
    Z, Y, X = x5d.shape[-3:]
    pad = (max(dx,0), max(-dx,0), max(dy,0),
           max(-dy,0), max(dz,0), max(-dz,0))
    x = F.pad(x5d, pad, mode='reflect')
    return x[:, :, max(-dz,0):max(-dz,0)+Z,
                  max(-dy,0):max(-dy,0)+Y,
                  max(-dx,0):max(-dx,0)+X]


def denoise_bilateral(y, sigma, delta=1.):
    """
    Bilateral denoising of a 2D or 3D image with additive Gaussian noise.
    Applies a non-linear filter combining:
    - spatial Gaussian weighting (σ_s ≈ 2.5·σ)
    - intensity similarity weighting (σ_r ≈ 2·σ)
    Preserves edges while reducing noise by averaging nearby pixels
    with similar intensities. Supports anisotropic 3D data via `delta`.

    Each output pixel x̂(i) is a weighted average of its neighbors:
        x̂(i) = (1 / Z(i)) · [sum over j of] w_s(i,j) · w_r(i,j) · y(j)
        where:
            w_s(i,j) = K(i - j)  (spatial Gaussian weight)
            w_r(i,j) = exp( -(y(i) - y(j))^2 / (2 * sigma_r^2) )  (intensity similarity Gaussian weight)
            Z(i) = [sum over j of] w_s(i,j) · w_r(i,j)  (normalization factor)
        and K is a normalized Gaussian kernel:
            K(j) ∝ exp( -||j||^2 / (2 * sigma_s^2) )
            sum_j K(j) = 1

    Args:
        y (Tensor): Input noisy image (1,Y,X) or (Z,Y,X)
        sigma (float): Noise standard deviation
        delta (float): Anisotropy ratio (Δz/Δxy) for 3D
    Returns:
        Tensor: Denoised image with same shape as input
    """
    y5d = _to_5d(y.to(device=device, dtype=dtype))
    is3d = _is_3d(y)
    sigma_s = 2.5 * sigma  # size of the neighborhood: [2-3]*sigma
    sigma_r = 2.0 * sigma  # pixel intensitiy torelance range: [1.5-2]*sigma
    K = (_kernel_gaussian(sigma_s, sigma_s/delta, True)
         if is3d else
         _kernel_gaussian(sigma_s, is3d=False).unsqueeze(0))
    # normalizing gaussian kernel:
    K /= K.sum()
    kZ, kY, kX = K.shape
    cz, cy, cx = kZ//2, kY//2, kX//2
    weighted_average = torch.zeros_like(y5d)  # numerator: sum_j w_s(i,j) · w_r(i,j) · y(j)
    Z = torch.zeros_like(y5d)  # normalization: Z(i) = sum_j w_s(i,j) · w_r(i,j)
    for iz in range(kZ):
        for iy in range(kY):
            for ix in range(kX):
                # offset j = (iz-cz, iy-cy, ix-cx)
                dz, dy, dx = iz - cz, iy - cy, ix - cx
                y_j = _shift_tensor(y5d, dz, dy, dx)   # y(j)
                w_s = K[iz, iy, ix]  # spatial weight K(j)
                # intensity similarity weight w_r(i,j), difference between y(j) and y(i):
                w_r = torch.exp(-((y_j - y5d) ** 2) / (2 * sigma_r ** 2))
                w = w_s * w_r  # total weight w_s(i,j) · w_r(i,j)
                weighted_average += w * y_j
                Z += w
    x5d = weighted_average / (Z + 1e-12)  # x̂(i) = (1/Z(i)) · sum_j w_s(i,j) · w_r(i,j) · y(j)
    return _from_5d(x5d)