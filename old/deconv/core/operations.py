import torch

import common.settings as settings


from common.utils import get_variables_from_dict

def build_gaussian_psf(sigma: float, kernel_size: int) -> torch.Tensor:
    """
    Build a 2D gaussian PSF (isotropic: sigma_x = sigma_y = sigma) with std sigma and kernel_size as kernel size.
    The PSF is normalised such that its sum equals 1.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # forcing the kernel size to be odd
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=settings.device, dtype=settings.dtype)
    y, x = torch.meshgrid(coords, coords, indexing='ij')
    psf = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    psf = psf / psf.sum()
    return psf


def compute_psf_from_params(psf_params: dict) -> torch.Tensor:
    """
    This function takes the measurement parameter dictionary 'psf_params' and returns the PSF kernel of the given
    parameters.
    """
    (sigma, kernel_size) = get_variables_from_dict(psf_params, ['sigma', 'kernel_size'])
    return build_gaussian_psf(sigma, kernel_size)


def _psf_to_fft_kernel(H: torch.Tensor, image_shape: tuple) -> torch.Tensor:
    """
    To compute the convolution operation, we will to a matrix multiplication in Fourier with torch.fft and torch.ifft:
    this function take H the PSF kernel and returns a padded version with the good shape in order to compute the
    Fourier Transform operations.
    """
    nY, nX = image_shape
    kH, kW = H.shape
    # zero padding with the PSF being centered to the left top corner:
    padded = torch.zeros((nY, nX), device=H.device, dtype=H.dtype)
    padded[:kH, :kW] = H
    # ifftshift to recenter the PSF on (0, 0):
    shift_y = -(kH // 2)
    shift_x = -(kW // 2)
    padded = torch.roll(padded, shifts=(shift_y, shift_x), dims=(0, 1))
    return padded


def apply_psf(H: torch.Tensor, f: torch.Tensor, adjoint=False) -> torch.Tensor:
    """
    Property: for a given convolution kernel H we have: conv(H, f) = A*f where A is the circulant matrix constructed
    from the kernel coefficients.

    This function computes the matrix multiplication A*f where A is the circulant PSF matrix and f a tensor from the
    mathematical set of the desired deconvoluted image.
    If adjoint=True this function computes the adjoint operation At*f where At is the conjugate transposed version of A.

    The matrix A is shape (nb of pix in f, nb of pix in f) and is intractable in memory, but because of the property of
    circulant matrix, the operation A*f is mathematically equivalent as fft(H)*fft(f) where H is the kernel version of
    the PSF
    In other term the inverse problem of deconvolution try to estimate A^{-1} but we work on H where:
    A = F*H*F with F the fft matrix operator, because all the operations are tractable from H which as a really lower
    dimension than A.
    """
    psf_fft_padded = _psf_to_fft_kernel(H, f.shape)
    f_fft = torch.fft.fft2(f)
    H_fft = torch.fft.fft2(psf_fft_padded)
    if adjoint:
        H_fft = H_fft.conj()
    return torch.real(torch.fft.ifft2(f_fft * H_fft))

