"""
Spatial differential operators for 2D and 3D images.

Central finite differences for gradient, divergence, laplacian, hessian.
GPU uses conv3d for parallelism; CPU uses slicing (conv3d is slow on CPU).
Supports anisotropic 3D via delta = dz / dxy.
When the image is 2D this module will use slicing both for GPU and CPU.
"""

import torch
import torch.nn.functional as F

import common.settings as settings


class DifferentialOperators:
    """Thread-safe differential operators — kernels are stored per instance."""

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(self, delta=1.):
        self.delta = delta
        if settings.device == 'cuda':
            self._build_3d_kernels()  # prebuild kernels if working on gpu

    # ── helpers ───────────────────────────────────────────────────────────

    ## pytorch conv3d needs 5-dimensional tensors
    @staticmethod
    def _to_5d(f): return f.unsqueeze(0).unsqueeze(0)
    @staticmethod
    def _from_5d(f): return f.squeeze(0).squeeze(0)
    @staticmethod
    def _is_3d(f): return f.dim() == 3

    # ── 3D GPU kernels (precomputed for conv3d) ──────────────────────────

    ## precomputes all conv3d kernels for first and second order derivatives
    def _build_3d_kernels(self):
        delta = self.delta
        def zero():
            return torch.zeros((3, 3, 3), device=settings.device, dtype=settings.dtype)
        # first order
        k_dz = zero()
        k_dz[2, 1, 1] = 1
        k_dz[0, 1, 1] = -1
        k_dz *= 0.5 * delta
        k_dy = zero()
        k_dy[1, 2, 1] = 1
        k_dy[1, 0, 1] = -1
        k_dy *= 0.5
        k_dx = zero()
        k_dx[1, 1, 2] = 1
        k_dx[1, 1, 0] = -1
        k_dx *= 0.5
        # second order diagonal
        k_dzz = zero()
        k_dzz[2, 1, 1] = 1
        k_dzz[1, 1, 1] = -2
        k_dzz[0, 1, 1] = 1
        k_dzz *= delta ** 2
        k_dyy = zero()
        k_dyy[1, 2, 1] = 1
        k_dyy[1, 1, 1] = -2
        k_dyy[1, 0, 1] = 1
        k_dxx = zero()
        k_dxx[1, 1, 2] = 1
        k_dxx[1, 1, 1] = -2
        k_dxx[1, 1, 0] = 1
        # second order mixed
        k_dyz = zero()
        k_dyz[0, 0, 1] = 1
        k_dyz[2, 2, 1] = 1
        k_dyz[0, 2, 1] = -1
        k_dyz[2, 0, 1] = -1
        k_dyz *= 0.25 * delta
        k_dxz = zero()
        k_dxz[0, 1, 0] = 1
        k_dxz[2, 1, 2] = 1
        k_dxz[0, 1, 2] = -1
        k_dxz[2, 1, 0] = -1
        k_dxz *= 0.25 * delta
        k_dxy = zero()
        k_dxy[1, 0, 0] = 1
        k_dxy[1, 2, 2] = 1
        k_dxy[1, 0, 2] = -1
        k_dxy[1, 2, 0] = -1
        k_dxy *= 0.25
        # pre-stacked kernels: (C_out, C_in, 3, 3, 3)
        self._grad_kernel = torch.stack([
            k_dz.unsqueeze(0),
            k_dy.unsqueeze(0),
            k_dx.unsqueeze(0)
        ], dim=0)
        self._div_kernel = torch.stack([
            (-k_dz),
            (-k_dy),
            (-k_dx)
        ], dim=0).unsqueeze(0)
        self._lap_kernel = (k_dzz + k_dyy + k_dxx).unsqueeze(0).unsqueeze(0)
        self._hess_kernel = torch.stack([
            k_dzz.unsqueeze(0),
            k_dyy.unsqueeze(0),
            k_dxx.unsqueeze(0),
            k_dyz.unsqueeze(0),
            k_dxz.unsqueeze(0),
            k_dxy.unsqueeze(0)
        ], dim=0)

    # ── 2D operators (slicing) ───────────────────────────────────────────

    def _spatial_grad_2d(self, f):
        padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1), mode='constant').squeeze(0).squeeze(0)
        dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
        dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
        return dy, dx

    def _divergence_2d(self, *components):
        fy, fx = components
        pad = lambda t: F.pad(t.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1), mode='constant').squeeze(0).squeeze(0)
        padded_y = pad(fy)
        padded_x = pad(fx)
        dy = (padded_y[2:, 1:-1] - padded_y[:-2, 1:-1]) * 0.5
        dx = (padded_x[1:-1, 2:] - padded_x[1:-1, :-2]) * 0.5
        return dy + dx

    def _laplacian_2d(self, f):
        padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1), mode='constant').squeeze(0).squeeze(0)
        dyy = padded[2:, 1:-1] - 2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]
        dxx = padded[1:-1, 2:] - 2 * padded[1:-1, 1:-1] + padded[1:-1, :-2]
        return dyy + dxx

    def _hessian_2d(self, f):
        padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1), mode='constant').squeeze(0).squeeze(0)
        dyy = padded[2:, 1:-1] - 2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]
        dxx = padded[1:-1, 2:] - 2 * padded[1:-1, 1:-1] + padded[1:-1, :-2]
        dxy = (padded[2:, 2:] + padded[:-2, :-2] - padded[2:, :-2] - padded[:-2, 2:]) * 0.25
        hess = torch.stack([dyy, dxy, dxy, dxx])
        return hess.view(2, 2, *f.shape)

    # ── 3D operators — CPU (slicing) ─────────────────────────────────────

    def _spatial_grad_3d_cpu(self, f):
        padded = self._from_5d(F.pad(self._to_5d(f), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
        dz = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1])
        dz *= 0.5 * self.delta
        dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
        dy *= 0.5
        dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
        dx *= 0.5
        return dz, dy, dx

    def _divergence_3d_cpu(self, *components):
        fz, fy, fx = components
        padded_z = self._from_5d(F.pad(self._to_5d(fz), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
        padded_y = self._from_5d(F.pad(self._to_5d(fy), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
        padded_x = self._from_5d(F.pad(self._to_5d(fx), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
        dz = (padded_z[2:, 1:-1, 1:-1] - padded_z[:-2, 1:-1, 1:-1])
        dz *= 0.5 * self.delta
        dy = (padded_y[1:-1, 2:, 1:-1] - padded_y[1:-1, :-2, 1:-1])
        dy *= 0.5
        dx = (padded_x[1:-1, 1:-1, 2:] - padded_x[1:-1, 1:-1, :-2])
        dx *= 0.5
        return dz + dy + dx

    def _laplacian_3d_cpu(self, f):
        padded = self._from_5d(F.pad(self._to_5d(f), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
        dzz = (padded[2:, 1:-1, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1])
        dzz *= self.delta**2
        dyy = padded[1:-1, 2:, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, :-2, 1:-1]
        dxx = padded[1:-1, 1:-1, 2:] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, 1:-1, :-2]
        return dzz + dyy + dxx

    def _hessian_3d_cpu(self, f):
        padded = self._from_5d(F.pad(self._to_5d(f), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
        # diagonal
        dzz = (padded[2:, 1:-1, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1])
        dzz *= self.delta**2
        dyy = (padded[1:-1, 2:, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, :-2, 1:-1])
        dxx = (padded[1:-1, 1:-1, 2:] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, 1:-1, :-2])
        # mixed
        dyz = padded[2:, 2:, 1:-1] + padded[:-2, :-2, 1:-1] - padded[2:, :-2, 1:-1] - padded[:-2, 2:, 1:-1]
        dyz *= 0.25 * self.delta
        dxz = padded[2:, 1:-1, 2:] + padded[:-2, 1:-1, :-2] - padded[2:, 1:-1, :-2] - padded[:-2, 1:-1, 2:]
        dxz *= 0.25 * self.delta
        dxy = padded[1:-1, 2:, 2:] + padded[1:-1, :-2, :-2] - padded[1:-1, 2:, :-2] - padded[1:-1, :-2, 2:]
        dxy *= 0.25
        hess = torch.stack([
            dzz, dyz, dxz,
            dyz, dyy, dxy,
            dxz, dxy, dxx
        ])
        return hess.view(3, 3, *f.shape)

    # ── 3D operators — GPU (conv3d) ──────────────────────────────────────

    def _spatial_grad_3d_gpu(self, f):
        out = F.conv3d(self._to_5d(f), self._grad_kernel, padding=1)
        out = out.squeeze(0)
        dz, dy, dx = out[0], out[1], out[2]
        return dz, dy, dx

    def _divergence_3d_gpu(self, *components):
        fz, fy, fx = components
        inp = torch.stack([fz, fy, fx], dim=0).unsqueeze(0)
        out = F.conv3d(inp, self._div_kernel, padding=1)
        return self._from_5d(out)

    def _laplacian_3d_gpu(self, f):
        out = F.conv3d(self._to_5d(f), self._lap_kernel, padding=1)
        return self._from_5d(out)

    def _hessian_3d_gpu(self, f):
        out = F.conv3d(self._to_5d(f), self._hess_kernel, padding=1).squeeze()
        out = out.squeeze(0)
        dzz, dyy, dxx, dyz, dxz, dxy = [self._from_5d(out[i]) for i in range(6)]
        hess = torch.stack([
            dzz, dyz, dxz,
            dyz, dyy, dxy,
            dxz, dxy, dxx
        ])
        return hess.view(3, 3, *f.shape)

    # ── public API ────────────────────────────────────────────────────────

    ## dispatches gradient based on f.dim() and device
    def spatial_grad(self, f):
        if not self._is_3d(f):
            return self._spatial_grad_2d(f)
        if settings.device == 'cuda':
            return self._spatial_grad_3d_gpu(f)
        return self._spatial_grad_3d_cpu(f)

    ## dispatches divergence based on number of components and device
    def divergence(self, *components):
        if len(components) == 2:
            return self._divergence_2d(*components)
        if settings.device == 'cuda':
            return self._divergence_3d_gpu(*components)
        return self._divergence_3d_cpu(*components)

    ## dispatches laplacian based on f.dim() and device
    def laplacian(self, f):
        if not self._is_3d(f):
            return self._laplacian_2d(f)
        if settings.device == 'cuda':
            return self._laplacian_3d_gpu(f)
        return self._laplacian_3d_cpu(f)

    ## dispatches hessian based on f.dim() and device
    def hessian(self, f):
        if not self._is_3d(f):
            return self._hessian_2d(f)
        if settings.device == 'cuda':
            return self._hessian_3d_gpu(f)
        return self._hessian_3d_cpu(f)
