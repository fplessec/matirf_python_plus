import torch

from .differential_operators import DifferentialOperators


class ProximalOperators:
    """
    Collection of proximal operators for 3D images, optimized for CPU and GPU.
    Loops over iterations (n_iter) are preserved, but all per-voxel operations
    are vectorized. Temporary tensors are preallocated and reused.
    """

    def __init__(self, delta=1.):
        self.diff_ops = DifferentialOperators(delta=delta)

    # pytorch conv need 5 dimensional tensors:
    @staticmethod
    def _to_5d(f): return f.unsqueeze(0).unsqueeze(0)
    @staticmethod
    def _from_5d(f): return f.squeeze(0).squeeze(0)

    # ----------------------------
    # SIMPLE PROX
    # ----------------------------
    def prox_positivity(self, f):
        """Hard positivity constraint: u >= 0"""
        return torch.clamp(f, min=0)

    def prox_l1(self, f, lambda_reg):
        """Soft-thresholding proximal for L1 norm"""
        return torch.sign(f) * torch.clamp(torch.abs(f) - lambda_reg, min=0)

    def prox_tikhonov(self, f, lambda_reg, tau=None, n_iter=50):
        """
        Proximal operator of L2 gradient norm (Tikhonov).
        Args:
            f: input 3D image [Z,Y,X]
            lambda_reg: regularization coefficient
            tau: step size for gradient descent ; default = 1 / (1 + 12*lambda_reg)
            n_iter: number of iterations
        """
        if tau is None:
            tau = 1 / (1 + 12 * lambda_reg)
        u = f.clone()
        for _ in range(n_iter):
            lap =  self.diff_ops.laplacian(u)
            grad = u - f - 2 * lambda_reg * lap
            u = u - tau * grad
        return u
        #
        # if tau is None:
        #     tau = 1 / (1 + 12 * lambda_reg)
        # u = f.clone()
        # lap = torch.empty_like(f)  # pre-allocated memory
        # for _ in range(n_iter):
        #     lap[:] = self.diff_ops.laplacian(u)  # avoid repeated allocations
        #     grad = u - f - 2 * lambda_reg * lap
        #     u = u - tau * grad
        # return u

    def prox_tikhonov_boulanger(self, f, lambda_reg, dt=10., max_iter=50):
        """
        Proximal operator of Tikhonov regularization (faithful to Boulanger's PDE).
        Minimizes: ||u - x||^2 + lambda_reg * ||grad(u)||^2
        Args:
            f: input 3D image [Z,Y,X]
            lambda_reg: regularization weight
            dt: base time step
            max_iter: number of iterations
        """
        u = f.clone()
        lap = torch.empty_like(f)  # pre-allocated memory
        for _ in range(max_iter):
            lap[:] = self.diff_ops.laplacian(u)  # avoid repeated allocations
            velocity = f - u + lambda_reg * lap
            v_min, v_max = velocity.min(), velocity.max()
            u = u + dt / (v_max - v_min + 1e-8) * velocity
        return u

    def prox_tv(self, f, lambda_reg, tau=0.125, n_iter=20):
        """
        Proximal Operator of the L1 norm of the gradient (Total Variation regularization) using Chambolle’s method
        Args:
            f: input 3D image [Z,Y,X]
            lambda_reg: regularization coefficient
            tau: dual step size (<= 1/6 for stability in 3D) ; default 1/8
            n_iter: number of Chambolle iterations (empirical 20–50)
        """
        u = f.clone()
        # pre-allocated memory to avoid repeated allocations in iterations:
        pz = torch.zeros_like(f)
        py = torch.zeros_like(f)
        px = torch.zeros_like(f)
        dz = torch.empty_like(f)
        dy = torch.empty_like(f)
        dx = torch.empty_like(f)
        div = torch.empty_like(f)
        for _ in range(n_iter):
            # primal update
            div[:] = self.diff_ops.divergence(pz, py, px)
            u[:] = f - lambda_reg * div
            # dual update
            dz[:], dy[:], dx[:] = self.diff_ops.spatial_grad(u)
            denom = 1.0 + tau * torch.sqrt(dz**2 + dy**2 + dx**2 + 1e-12)
            pz[:] = (pz + tau * dz) / denom
            py[:] = (py + tau * dy) / denom
            px[:] = (px + tau * dx) / denom
        div[:] = self.diff_ops.divergence(pz, py, px)
        return f - lambda_reg * div

    def prox_tv_normalized(self, f, lambda_reg, tau=0.125, n_iter=50):
        """
        Total Variation denoising (proximal Chambolle) in the style of Boulanger.
        Args:
            f (Tensor): input 3D image [Z,Y,X]
            lambda_reg (float): regularization weight
            tau (float): dual step size
            n_iter (int): number of Chambolle iterations
        """
        u = f.clone()
        # pre-allocated memory to avoid repeated allocations in iterations:
        pz = torch.zeros_like(f)
        py = torch.zeros_like(f)
        px = torch.zeros_like(f)
        dz = torch.empty_like(f)
        dy = torch.empty_like(f)
        dx = torch.empty_like(f)
        stacked = torch.empty((3, *f.shape), dtype=f.dtype, device=f.device)
        div = torch.empty_like(f)
        for _ in range(n_iter):
            # primal update
            div[:] = self.diff_ops.divergence(pz, py, px)
            u[:] = f - lambda_reg * div
            # dual update
            dz[:], dy[:], dx[:] = self.diff_ops.spatial_grad(u)
            stacked[0] = pz + tau * dz
            stacked[1] = py + tau * dy
            stacked[2] = px + tau * dx
            # normalization to constrain ||p|| ≤ 1:
            norm = torch.maximum(torch.ones_like(f), torch.sqrt((stacked**2).sum(dim=0)))
            pz[:] = stacked[0] / norm
            py[:] = stacked[1] / norm
            px[:] = stacked[2] / norm
        div[:] = self.diff_ops.divergence(pz, py, px)
        return f - lambda_reg * div


    def prox_hessian_frobenius(self, f, lambda_reg, step=0.1, n_iter=10):
        """
        Proximal of Frobenius norm of Hessian using gradient descent.

        step: gradient step size
        n_iter: number of iterations
        """
        u = f.clone()
        grad = torch.empty_like(f)  # pre-allocated memory
        for _ in range(n_iter):
            hess = self.diff_ops.hessian(u)
            # prox de la norme au carré de la hessienne:
            #grad[:] = hess.pow(2).sum(dim=(0, 1))  # avoid repeated allocations

            # prox de la norme de la hessienne (sqrt de la norme au carré):
            grad[:] = hess.sum(dim=(0, 1))  # avoid repeated allocations
            u = u - step * lambda_reg * grad
        return u

    def prox_shv(self, f, lambda_reg, rho=0.6, step=0.1, n_iter=10):
        """
        Proximal operator of the Sparse Hessian Variation (SHV).

        rho: weight between Hessian and L1 term
        step: gradient step
        n_iter: number of iterations
        """
        u = f.clone()
        grad = torch.empty_like(f)  # pre-allocated memory
        for _ in range(n_iter):
            hess = self.diff_ops.hessian(u)
            #grad_h = hess.pow(2).sum(dim=(0, 1))  voir comme juste au dessus, c pas la meme regularisation
            grad_h = hess.sum(dim=(0, 1))
            grad_l1 = torch.sign(u)
            grad[:] = rho * grad_h + (1 - rho) * grad_l1   # avoid repeated allocations
            u = u - step * lambda_reg * grad
        return u