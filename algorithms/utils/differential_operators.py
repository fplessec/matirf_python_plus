import torch
import torch.nn.functional as F

from settings import device, dtype


class DifferentialOperators:
    """
    This is a thread-safe class (e.g. kernels are stored per instance) that gather methods for computing differential
    operations for 3D volumes, optimized for CPU (uses fast finite differences) or GPU (uses 3D convolutions, better
    for parallelism).
    Considering that torch.conv3d can take quite some time when computed on CPU, that is why we choose two different
    styles of implementation for computing the differential operators, one for device='cpu' and the other one for
    device='cuda'.
    """

    def __init__(self, delta=1.,):
        self.delta = delta
        if device == 'cuda':
            self.build_kernels()

    # pytorch conv need 5 dimensional tensors:
    @staticmethod
    def _to_5d(f): return f.unsqueeze(0).unsqueeze(0)
    @staticmethod
    def _from_5d(f): return f.squeeze(0).squeeze(0)

    def build_kernels(self):
        """
        We will use the 1st-order central finite differences approximations of partial derivatives, encapsulated into
        3d convolution kernels for each first order derivative and second order partial derivatives.
        Those kernel will use delta to take into account the anisotropy of the 3d input image voxels.

        This function precompute all convolution kernels for first and second order derivatives.
        Kernels are stored in pre-stacked tensors to avoid runtime concatenations.
        """
        delta = self.delta  # anisotropy ratio = Δz / Δxy
        def zero():
            return torch.zeros((3, 3, 3), device=device, dtype=dtype)
        ###### first order
        # dz
        k_dz = zero()
        k_dz[2, 1, 1] = 1
        k_dz[0, 1, 1] = -1
        k_dz *= 0.5 * delta
        # dy
        k_dy = zero()
        k_dy[1, 2, 1] = 1
        k_dy[1, 0, 1] = -1
        k_dy *= 0.5
        # dx
        k_dx = zero()
        k_dx[1, 1, 2] = 1
        k_dx[1, 1, 0] = -1
        k_dx *= 0.5
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
        k_dyz[0, 0, 1] = 1
        k_dyz[2, 2, 1] = 1
        k_dyz[0, 2, 1] = -1
        k_dyz[2, 0, 1] = -1
        k_dyz *= 0.25 * delta
        # dxz
        k_dxz = zero()
        k_dxz[0, 1, 0] = 1
        k_dxz[2, 1, 2] = 1
        k_dxz[0, 1, 2] = -1
        k_dxz[2, 1, 0] = -1
        k_dxz *= 0.25 * delta
        # dxy
        k_dxy = zero()
        k_dxy[1, 0, 0] = 1
        k_dxy[1, 2, 2] = 1
        k_dxy[1, 0, 2] = -1
        k_dxy[1, 2, 0] = -1
        k_dxy *= 0.25
        ##### pre-stacked kernels: (C_out, C_in, 3, 3, 3)
        # for gradient:  (C_out=3, C_in=1, 3, 3, 3)
        self.grad_kernel = torch.stack([
            k_dz.unsqueeze(0),
            k_dy.unsqueeze(0),
            k_dx.unsqueeze(0)
        ], dim=0)
        # for divergence:  (C_out=1, C_in=3, 3, 3, 3)
        self.div_kernel = torch.stack([
            (-k_dz),
            (-k_dy),
            (-k_dx)
        ], dim=0).unsqueeze(0)
        # for laplacian:  (C_out=1, C_in=1, 3, 3, 3)
        self.lap_kernel = (k_dzz + k_dyy + k_dxx).unsqueeze(0).unsqueeze(0)
        # for hessian:  (C_out=6, C_in=1, 3, 3, 3)
        self.hess_kernel = torch.stack([
            k_dzz.unsqueeze(0),
            k_dyy.unsqueeze(0),
            k_dxx.unsqueeze(0),
            k_dyz.unsqueeze(0),
            k_dxz.unsqueeze(0),
            k_dxy.unsqueeze(0)
        ], dim=0)

    ############ Differential Operators:

    if device == 'cpu':
        ##### CPU Implementations:

        def spatial_grad(self, f):
            padded = self._from_5d(F.pad(self._to_5d(f), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
            # finite central difference computation: (first order approximation of the gradient)
            dz = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1])
            dz *= 0.5 * self.delta
            dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
            dy *= 0.5
            dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]
            dx *= 0.5
            return dz, dy, dx

        def divergence(self, z, y, x):
            padded_z = self._from_5d(F.pad(self._to_5d(z), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
            padded_y = self._from_5d(F.pad(self._to_5d(y), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
            padded_x = self._from_5d(F.pad(self._to_5d(x), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
            # finite central differences:
            dz = (padded_z[2:, 1:-1, 1:-1] - padded_z[:-2, 1:-1, 1:-1])
            dz *= 0.5 * self.delta
            dy = (padded_y[1:-1, 2:, 1:-1] - padded_y[1:-1, :-2, 1:-1])
            dy *= 0.5
            dx = (padded_x[1:-1, 1:-1, 2:] - padded_x[1:-1, 1:-1, :-2])
            dx *= 0.5
            # sum of derivatives -> divergence:
            div = dz + dy + dx
            return div

        def laplacian(self, f):
            padded = self._from_5d(F.pad(self._to_5d(f), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
            # diagonal second-order central finite differences:
            dzz = (padded[2:, 1:-1, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1])
            dzz *= self.delta**2
            dyy = padded[1:-1, 2:, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, :-2, 1:-1]
            dxx = padded[1:-1, 1:-1, 2:] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, 1:-1, :-2]
            # sum of second derivatives -> laplacian:
            lap = dzz + dyy + dxx
            return lap

        def hessian(self, f):
            padded = self._from_5d(F.pad(self._to_5d(f), pad=(1, 1, 1, 1, 1, 1), mode='constant'))
            # diagonal second-order central finite differences:
            dzz = (padded[2:, 1:-1, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1])
            dzz *= self.delta**2
            dyy = (padded[1:-1, 2:, 1:-1] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, :-2, 1:-1])
            dxx = (padded[1:-1, 1:-1, 2:] - 2 * padded[1:-1, 1:-1, 1:-1] + padded[1:-1, 1:-1, :-2])
            # cross second-order central finite differences:
            dyz = padded[2:, 2:, 1:-1] + padded[:-2, :-2, 1:-1] - padded[2:, :-2, 1:-1] - padded[:-2, 2:, 1:-1]
            dyz *= 0.25 * self.delta
            dxz = padded[2:, 1:-1, 2:] + padded[:-2, 1:-1, :-2] - padded[2:, 1:-1, :-2] - padded[:-2, 1:-1, 2:]
            dxz *= 0.25 * self.delta
            dxy = padded[1:-1, 2:, 2:] + padded[1:-1, :-2, :-2] - padded[1:-1, 2:, :-2] - padded[1:-1, :-2, 2:]
            dxy *= 0.25
            # stack into hessian matrix:
            hess = torch.stack([
                dzz, dyz, dxz,
                dyz, dyy, dxy,
                dxz, dxy, dxx
            ])
            return hess.view(3, 3, *f.shape)


    elif device == 'cuda':
        ##### GPU Implementations:

        def spatial_grad(self, f):
            # conv3d( input(1,1,D,H,W), grad_kernel(1,3,3,3,3) ) -> output(1,3,D,H,W)
            out = F.conv3d(self._to_5d(f), self.grad_kernel, padding=1)
            # out.shape = (1, 3, D, H, W)
            out = out.squeeze(0)  # (3, D, H, W)
            dz, dy, dx = out[0], out[1], out[2]
            return dz, dy, dx  # (D, H, W)

        def divergence(self, z, y, x):
            # 5d input the field (z,y,x):
            inp = torch.stack([z, y, x], dim=0).unsqueeze(0)  # (1,3,D,H,W)
            # conv3d( input(1,3,D,H,W), div_kernel(1,3,3,3,3) ) -> output(1,1,D,H,W)
            out = F.conv3d(inp, self.div_kernel, padding=1)  # (1,1,D,H,W)
            return self._from_5d(out)  # (D,H,W)

        def laplacian(self, f):
            # conv3d( input(1,1,D,H,W), lap_kernel(1,1,3,3,3) ) -> output(1,1,D,H,W)
            out = F.conv3d(self._to_5d(f), self.lap_kernel, padding=1)  # (1,1,D,H,W)
            return self._from_5d(out)  # (D,H,W)

        def hessian(self, f):
            # conv3d( input(1,1,D,H,W), hess_kernel(6,1,3,3,3) ) -> output(1,6,D,H,W)
            out = F.conv3d(self._to_5d(f), self.hess_kernel, padding=1).squeeze()
            # out.shape = (1, 6, D, H, W)
            out = out.squeeze(0)  # (6, D, H, W)
            dzz, dyy, dxx, dyz, dxz, dxy = [self._from_5d(out[i]) for i in range(6)]
            hess = torch.stack([
                dzz, dyz, dxz,
                dyz, dyy, dxy,
                dxz, dxy, dxx
            ])
            return hess.view(3, 3, *f.shape)  # (3, 3, D, H, W)
