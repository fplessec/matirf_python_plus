import torch
import torch.nn.functional as F

from settings import device, dtype


class DifferentialOperators:
    """
    Differential operators for 3D volumes, optimized for CPU or GPU:
    - CPU: uses fast finite differences (vectorized)
    - GPU: uses 3D convolutions (better for parallelism)
    - Thread-safe: kernels are stored per instance
    """

    def __init__(self, delta=1.):
        self.delta = delta
        if device == 'gpu':
            self.build_kernels()

    def build_kernels(self):
        """
        We will use the 1st-order central finite differences approximations of partial derivatives, encapsulated into
        3d convolution kernels for each first order derivative and second order partial derivatives.
        Those kernel will use delta to take into account the anisotropy of the 3d input image voxels.

        This function precompute all convolution kernels for first and second order derivatives.
        Kernels are stored in pre-stacked tensors to avoid runtime concatenations.
        """
        delta = self.delta
        def zero():
            return torch.zeros((3, 3, 3), device=device, dtype=dtype)
        ###### first order
        # dz
        k_dz = zero()
        k_dz[2, 1, 1] = 1
        k_dz[0, 1, 1] = -1
        k_dz *= delta
        # dy
        k_dy = zero()
        k_dy[1, 2, 1] = 1
        k_dy[1, 0, 1] = -1
        # dx
        k_dx = zero()
        k_dx[1, 1, 2] = 1
        k_dx[1, 1, 0] = -1
        ##### second order diagonal
        # dzz
        k_dzz = zero()
        k_dzz[2, 1, 1] = 1
        k_dzz[1, 1, 1] = -2
        k_dzz[0, 1, 1] = 1
        k_dzz *= delta ** 2
        # dyy
        k_dyy = zero()
        k_dyy[1, 2, 1] = 1
        k_dyy[1, 1, 1] = -2
        k_dyy[1, 0, 1] = 1
        # dxx
        k_dxx = zero()
        k_dxx[1, 1, 2] = 1
        k_dxx[1, 1, 1] = -2
        k_dxx[1, 1, 0] = 1
        ##### second order mixed
        # dyz
        k_dyz = zero()
        k_dyz[2, 2, 1] = 1
        k_dyz[0, 0, 1] = 1
        k_dyz[2, 0, 1] = -1
        k_dyz[0, 2, 1] = -1
        k_dyz *= 0.25 * delta
        # dxz
        k_dxz = zero()
        k_dxz[2, 1, 2] = 1
        k_dxz[0, 1, 0] = 1
        k_dxz[2, 1, 0] = -1
        k_dxz[0, 1, 2] = -1
        k_dxz *= 0.25 * delta
        # dxy
        k_dxy = zero()
        k_dxy[1, 2, 2] = 1
        k_dxy[1, 0, 0] = 1
        k_dxy[1, 2, 0] = -1
        k_dxy[1, 0, 2] = -1
        k_dxy *= 0.25

        ##### pre-stacked kernels
        # for gradient:
        self.grad_kernel = torch.stack([
            k_dx.unsqueeze(0),
            k_dy.unsqueeze(0),
            k_dz.unsqueeze(0)
        ], dim=0)
        print("self.grad_kernel.shape", self.grad_kernel.shape)
        # for divergence:
        self.div_kernel = torch.stack([
            (-k_dz).unsqueeze(0),
            (-k_dy).unsqueeze(0),
            (-k_dx).unsqueeze(0)
        ], dim=0)
        print("self.div_kernel.shape", self.div_kernel.shape)
        # for laplacian:
        self.lap_kernel = torch.stack([
            k_dxx.unsqueeze(0),
            k_dyy.unsqueeze(0),
            k_dzz.unsqueeze(0)
        ], dim=0)
        print("self.lap_kernel.shape", self.lap_kernel.shape)
        # for hessian:
        self.hessian_kernel = torch.stack([
            k_dxx.unsqueeze(0),
            k_dyy.unsqueeze(0),
            k_dzz.unsqueeze(0),
            k_dxy.unsqueeze(0),
            k_dxz.unsqueeze(0),
            k_dyz.unsqueeze(0)
        ], dim=0)
        print("self.hessian_kernel.shape", self.hessian_kernel.shape)

    # pytorch conv need 5 dimensional tensors:
    @staticmethod
    def _to_5d(f): return f.unsqueeze(0).unsqueeze(0)
    @staticmethod
    def _from_5d(f): return f.squeeze(0).squeeze(0)


    ############ Differential Operators:

    if device == 'cpu':
        ##### CPU Implementations:

        def spatial_grad(self, f):
            padded = F.pad(f.unsqueeze(0).unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
            # finite difference computation: (first order approximation of the gradient)
            dz = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) * self.delta
            dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
            dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
            return dz, dy, dx
            # f5 = f.unsqueeze(0).unsqueeze(0)
            # padded = F.pad(f5, (1, 1, 1, 1, 1, 1), mode='replicate').squeeze()
            # dz = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) * self.delta
            # dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
            # dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
            # return dz, dy, dx
            # padded = self._from_5d(F.pad(self._to_5d(f), (1, 1, 1, 1, 1, 1), mode='replicate'))
            # dz = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) * self.delta
            # dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
            # dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
            # return dz, dy, dx
            #
            # return dx, dy, dz

        def divergence(self, z, y, x):
            div = torch.zeros_like(z)
            # z direction:
            div[1:-1, :, :] += z[1:-1, :, :] - z[:-2, :, :]
            div[0, :, :] += z[0, :, :]
            div[-1, :, :] -= z[-2, :, :]
            # y direction:
            div[:, 1:-1, :] += y[:, 1:-1, :] - y[:, :-2, :]
            div[:, 0, :] += y[:, 0, :]
            div[:, -1, :] -= y[:, -2, :]
            # x direction:
            div[:, :, 1:-1] += x[:, :, 1:-1] - x[:, :, :-2]
            div[:, :, 0] += x[:, :, 0]
            div[:, :, -1] -= x[:, :, -2]
            return div
            # div = torch.zeros_like(z)
            # div[1:-1] += z[1:-1] - z[:-2]
            # div[0] += z[0]
            # div[-1] -= z[-2]
            # div[:, 1:-1] += y[:, 1:-1] - y[:, :-2]
            # div[:, 0] += y[:, 0]
            # div[:, -1] -= y[:, -2]
            # div[:, :, 1:-1] += x[:, :, 1:-1] - x[:, :, :-2]
            # div[:, :, 0] += x[:, :, 0]
            # div[:, :, -1] -= x[:, :, -2]
            # return div
            # dz_pad = self._from_5d(F.pad(self._to_5d(z), (0, 0, 0, 0, 1, 1), mode='replicate'))
            # dy_pad = self._from_5d(F.pad(self._to_5d(y), (0, 0, 1, 1, 0, 0), mode='replicate'))
            # dx_pad = self._from_5d(F.pad(self._to_5d(x), (1, 1, 0, 0, 0, 0), mode='replicate'))
            # div = ((dz_pad[2:] - dz_pad[:-2]) * self.delta +
            #        dy_pad[:, 2:] - dy_pad[:, :-2] +
            #        dx_pad[:, :, 2:] - dx_pad[:, :, :-2])
            # return div

        def laplacian(self, f):
            dz, dy, dx = self.spatial_grad(f)
            return self.divergence(dz, dy, dx)
            # f5 = f.unsqueeze(0).unsqueeze(0)
            # padded = F.pad(f5, (1, 1, 1, 1, 1, 1), mode='replicate').squeeze()
            #
            # center = padded[1:-1, 1:-1, 1:-1]
            #
            # lap = (
            #         (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1] - 2 * center) * self.delta ** 2
            #         + (padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1] - 2 * center)
            #         + (padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2] - 2 * center)
            # )
            #
            # return lap
            # dz, dy, dx = self.spatial_grad(f)
            # return self.divergence(dz, dy, dx)
            #
            # dz, dy, dx = self.spatial_grad(f)
            # ddx, ddy, ddz = self.spatial_grad(dx)[0], self.spatial_grad(dy)[1], self.spatial_grad(dz)[2]
            # return ddx + ddy + ddz

        def hessian(self, f):
            dz, dy, dx = self.spatial_grad(f)
            dzdz, dzdy, dzdx = self.spatial_grad(dz)
            dydz, dydy, dydx = self.spatial_grad(dy)
            dxdz, dxdy, dxdx = self.spatial_grad(dx)
            hess = torch.stack([
                dzdz, dzdy, dzdx,
                dydz, dydy, dydx,
                dxdz, dxdy, dxdx
            ])
            return hess.view(3, 3, *f.shape)


    elif device == 'gpu':
        ##### GPU Implementations:

        def spatial_grad(self, f):
            out = F.conv3d(self._to_5d(f), self.grad_kernel, padding=1)
            # out.shape = (1, 3, D, H, W)
            out = out.squeeze(0)  # (3, D, H, W)
            dx, dy, dz = out[0], out[1], out[2]
            return dx, dy, dz

        def divergence(self, z, y, x):
            inp = torch.stack([z, y, x]).unsqueeze(1)  # (3,1,D,H,W)
            out = F.conv3d(inp, self.div_kernel, padding=1, groups=3)
            return self._from_5d(out.sum(0))

        def laplacian(self, f):
            out = F.conv3d(self._to_5d(f), self.lap_kernel, padding=1)
            return self._from_5d(out.sum(0))

        def hessian(self, f):
            out = F.conv3d(self._to_5d(f), self.hessian_kernel, padding=1)
            dxx, dyy, dzz, dxy, dxz, dyz = [self._from_5d(out[i]) for i in range(6)]
            H = torch.stack([dzz, dyz, dxz, dyz, dyy, dxy, dxz, dxy, dxx])
            return H.view(3, 3, *f.shape)