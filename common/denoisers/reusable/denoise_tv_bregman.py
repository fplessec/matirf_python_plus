"""
total-variation denoising using split-Bregman optimization

The denoised image will be returned as type torch.Tensor
Should be convert to numpy array for plotting
"""

__author__ = "Yi-Tang Wang"
__email__ = "yitang.wang@uq.net.au"
__reference__ = ["skimage.restoration.denoise_tv_bregman"]

import torch
import math

from common.denoisers.base import Denoiser


def _denoise_tv_bregman(image, weight, max_iter=100, eps=1e-3):
    """Perform total-variation denoising using split-Bregman optimization.

    Parameters:
        image (torch.Tensor):
            Input data to be denoised.
        weight (float):
            Denoising weight. The smaller the 'weight', the more denoising (at
            the expense of less similarity to the 'input').
        max_iter (int):
            Optional
            Maximal number of iterations used for the optimization.
        eps (float):
            Optional
            The threshold of distance between denoised image in iterations
            The algorithm stops when image distance is smaller than eps

    Returns:
        out (torch.Tensor): denoised image
    """
    image = atleast_3d(image)

    img_shape = list(image.shape)
    rows = img_shape[0]
    rows2 = rows + 2
    cols = img_shape[1]
    cols2 = cols + 2
    dims = img_shape[2]
    total = rows * cols * dims
    shape_extend = (rows2, cols2, dims)
    # out is firstly created as zeros-like tensor with size as shape_extend
    out = torch.zeros(shape_extend, dtype=torch.float)

    dx = out.clone().detach()
    dy = out.clone().detach()
    bx = out.clone().detach()
    by = out.clone().detach()

    lam = 2 * weight
    rmse = float("inf")
    norm = (weight + 4 * lam)

    out_rows, out_cols = out.shape[:2]
    out[1:out_rows-1, 1:out_cols-1] = image

    out = fill_extend(image, out)

    i = 0
    regularization = torch.mul(image, weight)
    # iterative optimization method
    # split-Bregman iteration
    while i < max_iter and rmse > eps:
        uprev = out[1:-1, 1:-1, :]

        ux = out[1:-1, 2:, :] - uprev
        uy = out[2:, 1:-1, :] - uprev

        unew = torch.div(
            (torch.mul((out[2:, 1:-1, :]
                    + out[0:-2, 1:-1, :]
                    + out[1:-1, 2:, :]
                    + out[1:-1, 0:-2, :]

                    + dx[1:-1, 0:-2, :]
                    - dx[1:-1, 1:-1, :]
                    + dy[0:-2, 1:-1, :]
                    - dy[1:-1, 1:-1, :]

                    - bx[1:-1, 0:-2, :]
                    + bx[1:-1, 1:-1, :]
                    - by[0:-2, 1:-1, :]
                    + by[1:-1, 1:-1, :]), lam) + regularization),
             norm)
        out[1:-1, 1:-1, :] = unew.clone().detach()

        rmse = torch.norm(unew-uprev, p=2)

        bxx = bx[1:-1, 1:-1, :].clone().detach()
        byy = by[1:-1, 1:-1, :].clone().detach()

        tx = ux + bxx
        ty = uy + byy
        s = torch.sqrt(torch.pow(tx, 2)+torch.pow(ty, 2))
        dxx = torch.div(torch.addcmul(torch.zeros(s.shape, dtype=torch.float), lam, s, tx),
                        torch.add(torch.mul(s, lam), 1))
        dyy = torch.div(torch.addcmul(torch.zeros(s.shape, dtype=torch.float), lam, s, ty),
                        torch.add(torch.mul(s, lam), 1))

        dx[1:-1, 1:-1, :] = dxx.clone().detach()
        dy[1:-1, 1:-1, :] = dyy.clone().detach()

        bx[1:-1, 1:-1, :] += ux - dxx
        by[1:-1, 1:-1, :] += uy - dyy

        i += 1
    # return the denoised image excluding the extended area
    return out[1:-1, 1:-1]


def atleast_3d(image):
    """to ensure the image has at least 3 dimensions

    if the input image already has at least 3 dimensions, just return the image
    otherwise, extend the dimensionality of the image to 3 dimensions

    Parameters:
        image (torch.Tensor):
            input image

    Return:
        image (torch.Tensor):
            image that has at least 3 dimensions
    """
    dim = list(image.shape)

    if len(dim) >= 3:
        return image
    else:
        dim.append(1)
        return image.view(dim)


def fill_extend(image, out):
    """fill the extended area in out img with original img"""
    out_rows, out_cols = out.shape[:2]
    rows, cols = out_rows - 2, out_cols - 2
    out[0, 1:out_cols-1] = image[1, :]
    out[1:out_rows-1, 0] = image[:, 1]
    out[out_rows-1, 1:out_cols-1] = image[rows-1, :]
    out[1:out_rows-1, out_cols-1] = image[:, cols-1]
    return out


class TVBregmanDenoiser(Denoiser):

    name = "TV Bregman"
    supports_3d = False
    supports_anisotropy = False

    def __init__(self, max_iter=100, eps=1e-3):
        self.max_iter = max_iter
        self.eps = eps

    def denoise(self, y, sigma, delta=1.0):
        """Adapter: TV Bregman uses 'weight' instead of 'sigma'.
        weight = 1/sigma gives stronger denoising for larger noise."""
        weight = max(1.0 / sigma, 0.1) if sigma > 0 else 10.0
        # y is (1, Y, X) — squeeze to (Y, X) for the 2D implementation
        image_2d = y.squeeze(0)
        result = _denoise_tv_bregman(image_2d, weight=weight,
                                     max_iter=self.max_iter, eps=self.eps)
        return result.unsqueeze(0)


if __name__=="__main__":

    from common.in_out import load_tif

    g = load_tif("esoubies.TIF")
    print(g.shape)
    print(g.min().item(), g.max().item())
    g = (g - g.min()) / (g.max() - g.min()) * 255

    print(g.min().item(), g.max().item())

    sigma = 0.1
    sigma = 20
    noise = torch.randn_like(g) * sigma
    g_noisy = g + noise


    from skimage.restoration import estimate_sigma
    sigma_est = estimate_sigma(g_noisy, average_sigmas=True, channel_axis=0)
    print(sigma_est)

    denoiser = TVBregmanDenoiser()
    g_denoised = denoiser(g_noisy, sigma=sigma_est)

    diff = (g - g_denoised).pow(2).sqrt()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(221)
    plt.imshow(g[0,:,:].numpy())
    plt.title("true")
    plt.subplot(222)
    plt.imshow(g_noisy[0,:,:].numpy())
    plt.title("noisy")
    plt.subplot(223)
    plt.imshow(g_denoised[0,:,:].numpy())
    plt.title("denoised")
    plt.subplot(224)
    plt.imshow(diff[0,:,:].numpy())
    plt.show()
