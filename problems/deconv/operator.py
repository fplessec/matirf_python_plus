"""
The deconvolution forward operator: blurring by a point-spread function.

H convolves the image with a PSF. Written as a matrix it would be circulant of size
(n_pixels, n_pixels) — completely intractable. But a circulant matrix is diagonal in the
Fourier basis, so the whole operator reduces to a pointwise multiplication by the PSF's
spectrum:

    H f      = F^-1[ F(f) * Hhat ]
    H^T y    = F^-1[ F(y) * conj(Hhat) ]

This is the counterpart of MA-TIRF: there H was a small explicit matrix, here H is never
formed at all. The same six solvers drive both without noticing the difference — which is
the point of putting `apply` / `adjoint` behind an interface.

The closed form that matters
----------------------------
Because H is diagonal in Fourier, so is H^T H, and the regularized normal equations become
a scalar division per frequency:

    (H^T H + lam I)^-1 b  =  F^-1[ F(b) / (|Hhat|^2 + lam) ]

That is the Wiener filter, and it is exact. Overriding `solve_normal` with it turns the
data step of PPXA, ADMM, PnP and PnP-ADMM — and `ridge_inverse`, hence MCMC — from an
iterative conjugate gradient into one FFT round trip.
"""

import torch

import common.settings as settings
from core import Feature, ForwardOperator, features


def gaussian_psf(sigma: float, kernel_size: int) -> torch.Tensor:
    """
    An isotropic 2D Gaussian PSF, normalized to sum to 1.

    Summing to 1 makes the blur preserve total intensity, so a flat image blurs to itself.
    The kernel size is forced odd so the PSF has a well-defined centre pixel.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=settings.device, dtype=settings.dtype)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    psf = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return psf / psf.sum()


class DeconvOperator(ForwardOperator):
    """Convolution by a PSF, evaluated in Fourier. H is never formed explicitly."""

    name = "deconv"
    features = features(Feature.TWO_D)

    def __init__(self, psf: torch.Tensor, psf_params: dict = None):
        self.psf = psf
        self.psf_params = psf_params or {}
        self._spectra = {}          # image shape -> the PSF's spectrum at that shape

    @classmethod
    def from_config(cls, config: dict) -> "DeconvOperator":
        from common.in_out import load_json

        params = load_json(config["input-paths"]["json"])
        return cls(gaussian_psf(params["sigma"], params["kernel_size"]), params)

    # ── the PSF's spectrum, cached per image shape ───────────────────────────

    def spectrum(self, shape) -> torch.Tensor:
        """
        F(psf), zero-padded to `shape` and rolled so the PSF's centre sits at index (0, 0).

        The roll matters: a convolution kernel must be centred on the origin in the Fourier
        convention, otherwise the result comes back shifted by half the kernel.

        Cached because every solver calls this once per iteration with the same shape.
        """
        shape = tuple(shape)
        cached = self._spectra.get(shape)
        if cached is not None:
            return cached
        kh, kw = self.psf.shape
        padded = torch.zeros(shape, device=self.psf.device, dtype=self.psf.dtype)
        padded[:kh, :kw] = self.psf
        padded = torch.roll(padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
        spectrum = torch.fft.fft2(padded)
        self._spectra[shape] = spectrum
        return spectrum

    # ── the physics ──────────────────────────────────────────────────────────

    def apply(self, f: torch.Tensor) -> torch.Tensor:
        """H f — blur the image."""
        return torch.real(torch.fft.ifft2(torch.fft.fft2(f) * self.spectrum(f.shape)))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """H^T y — correlate with the PSF (the conjugate spectrum)."""
        return torch.real(torch.fft.ifft2(torch.fft.fft2(y) * self.spectrum(y.shape).conj()))

    # ── the Wiener closed form ───────────────────────────────────────────────

    def solve_normal(self, b: torch.Tensor, lam: float = 0.0, **kwargs) -> torch.Tensor:
        """
        (H^T H + lam I)^-1 b, as one pointwise division in Fourier.

        With lam = 0 this is the naive inverse filter, and a Gaussian PSF's spectrum decays
        to (numerically) zero at high frequency: dividing by it amplifies noise without
        bound. The denominator is floored at a tiny epsilon so the call cannot produce
        infinities, but a meaningful lam > 0 is what actually makes the inversion stable —
        that is the whole reason `ridge_inverse` takes one.
        """
        denominator = self.spectrum(b.shape).abs() ** 2 + lam
        denominator = denominator.clamp(min=torch.finfo(self.psf.dtype).tiny)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(b) / denominator))

    # ── a capability of the physics, which the GUI reads ─────────────────────

    def spectrum_magnitudes(self, shape) -> torch.Tensor:
        """
        |Hhat| for every frequency, sorted descending.

        The GUI's "Estimate lambda_rr" dialog plots these and lets the user pick a cutoff:
        below a given magnitude, a frequency is drowned in noise and should be damped. In
        v1 this lived in `deconv/gui/` and was imported by `deconv/algorithms/`, which is
        what made the algorithm layer depend on the GUI. It is a property of the PSF.
        """
        return torch.sort(self.spectrum(shape).abs().flatten(), descending=True).values

    def __repr__(self) -> str:
        return f"<DeconvOperator psf={tuple(self.psf.shape)} params={self.psf_params}>"
