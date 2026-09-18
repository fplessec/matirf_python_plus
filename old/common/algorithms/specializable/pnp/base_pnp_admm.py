"""
Plug-and-Play ADMM (PnP-ADMM) for inverse problems.

This is the augmented-Lagrangian counterpart of BasePnp (which is plain HQS):
it solves the same reconstruction but keeps a dual variable u, which makes the
denoiser prior act as an exact operator splitting and converges robustly for a
*fixed* penalty rho (no Kai-Zhang annealing required).

Solves  min_x  1/2 ||H x - g||^2 + phi(v)   subject to   x = v,
with phi the implicit prior encoded by a denoiser (Venkatakrishnan et al. 2013).
Scaled-form ADMM iterations:
    x-step (data)  :  x = (H^t H + rho I)^{-1} (H^t g + rho (v - u))
    v-step (prior) :  v = D_sigma(x + u)          <- denoiser replaces a prox
    u-step (dual)  :  u = u + (x - v)

Scale bridge:
    x lives in [0, 1] but the denoisers are calibrated for the [0, 255] intensity
    convention, so the prior step denoises ``D((x+u) * 255, sigma) / 255`` with
    sigma expressed on that 255 scale (see BasePnp for details).

Subclasses must override:
    apply_forward(H, f)                     : computes Hf
    apply_adjoint(H, x)                     : computes H^t x
    solve_linear_system(Htg, v, rho, params): solves (H^t H + rho I)^{-1} (Htg + rho v)
    init_x(g, H, params)                    : initialization of x
Subclasses may override:
    precompute(g, H, params)                : cache operators reused every iteration
"""

from typing import Dict, Any
import time

import torch

from common.algorithms.base import Algorithm
from common.algorithms.specializable.pnp.ui_params_admm import PNP_ADMM_UI_PARAMETERS
from common.denoisers import DENOISER_REGISTRY
from common.utils import get_variables_from_dict


## denoisers are calibrated for the [0, 255] intensity convention:
_DENOISER_SCALE = 255.0


class BasePnpAdmm(Algorithm):

    name = "ADMM-PnP"
    ui_params = PNP_ADMM_UI_PARAMETERS
    estimator_type = "MAP"
    uses_denoiser = True
    uses_regularization = False

    def run(self, g, H, params: Dict[str, Any]):
        self.fix_randomness()

        (n_iter, rho, sigma, denoiser, forced_pos) = get_variables_from_dict(
            params, ['iter', 'rho', 'sigma', 'denoiser', 'forced_pos'])
        ## delta is only present for anisotropic problems (matirf); default isotropic:
        delta = params.get('delta', 1.0)

        self.precompute(g, H, params)

        Htg = self.apply_adjoint(H, g)
        x = self.init_x(g, H, params)
        v = x.clone()
        u = torch.zeros_like(x)

        self._warn_if_slice_by_slice(denoiser, x)

        denoiser_fn = None if denoiser == "None" else DENOISER_REGISTRY.get(denoiser)
        if denoiser != "None" and denoiser_fn is None:
            self._print(f"denoiser '{denoiser}' not found in registry, stopping")
            return None

        t0 = time.time()

        for k in range(n_iter):
            if self.is_stop_requested():
                return None
            self._print(f"[ADMM-PnP] iter {k + 1}/{n_iter} | rho = {rho:.3g} | sigma = {sigma:.3g}")

            # x-step (data): (H^t H + rho I)^{-1} (H^t g + rho (v - u)):
            x = self.solve_linear_system(Htg, v - u, rho, params)

            # v-step (prior): denoiser on the native [0, 255] scale:
            if denoiser_fn is None:
                v = x + u
            else:
                v = self._denoise(denoiser_fn, x + u, sigma, delta)
            if forced_pos:
                v = v.clamp(min=0)

            # u-step (dual):
            u = u + (x - v)

            self._update_figure(v)

        self._print(f"execution in {time.time() - t0:.2f} s")
        return v

    @staticmethod
    def _denoise(denoiser_fn, f, sigma, delta):
        """Denoises f on the denoiser's native [0, 255] scale, back to f's scale."""
        return (denoiser_fn(f * _DENOISER_SCALE, sigma, delta) / _DENOISER_SCALE).reshape(f.shape)

    def precompute(self, g, H, params):
        """Precomputes operators reused by solve_linear_system. Store as self attributes."""
        pass

    def solve_linear_system(self, Htg, v, rho, params):
        """Solves (H^t H + rho I)^{-1} (Htg + rho v). Must be overridden by subclasses."""
        raise NotImplementedError

    def init_x(self, g, H, params):
        """Initialization of x. Default: the back-projection H^t g clamped to positive."""
        return self.apply_adjoint(H, g).clamp(min=0)
