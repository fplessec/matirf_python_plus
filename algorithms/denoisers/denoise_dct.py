import torch

from settings import device, dtype


def _dct_matrix(N):
    n = torch.arange(N, device=device, dtype=dtype).view(1, -1)
    k = torch.arange(N, device=device, dtype=dtype).view(-1, 1)
    M = torch.cos(torch.pi / N * (n + 0.5) * k)
    M[0] *= 1.0 / (2. ** 0.5)
    return M * (2. / N) ** 0.5

def _dct_nd(x):
    X = x.clone()
    for axis in range(x.dim()):
        D = _dct_matrix(x.shape[axis])
        X = torch.tensordot(D, X, dims=([1],[axis]))
    return X

def _idct_nd(X):
    x = X.clone()
    for axis in range(X.dim()):
        D = _dct_matrix(X.shape[axis])
        x = torch.tensordot(D.t(), x, dims=([1],[axis]))
    return x

def denoise_dct(y, sigma):
    """
    DCT-based denoising of an image with additive Gaussian noise.
    Applies an orthonormal Discrete Cosine Transform (DCT), performs
    soft-thresholding in the transform domain, and reconstructs the image.

    The denoised signal is obtained as:
        x̂ = IDCT(X̂)
    where each coefficient is shrunk independently:
        X̂(k) = sign(X(k)) · max(|X(k)| - T, 0)
        with:
            X = DCT(y)
            T = sigma · sqrt(2 · log(N))   (universal threshold)
            N = total number of elements

    Args:
        y (Tensor): Input noisy image (any dimension)
        sigma (float): Noise standard deviation
    Returns:
        Tensor: Denoised image with same shape as input
    """
    y = y.to(device=device, dtype=dtype)
    Y = _dct_nd(y)
    N = y.numel()
    T = sigma * torch.sqrt(2 * torch.log(torch.tensor(N, device=device, dtype=dtype)))
    Xh = torch.sign(Y) * torch.clamp(torch.abs(Y) - T, min=0.)
    return _idct_nd(Xh)