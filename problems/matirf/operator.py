"""
The MA-TIRF forward operator.

H is a small dense matrix of shape (n_angles, nz): one row per incidence angle, one column
per reconstructed depth slice. Applying it mixes the z slices of f into the measured stacks.

Two things make this operator worth reading as an example of the v2 contract:

    > It overrides `solve_normal`. H is small, so (H^T H + lam I) can be inverted directly
      instead of by conjugate gradient. Because every proximal solver's data step and
      `ridge_inverse` are both built on `solve_normal`, that ONE override makes PPXA, ADMM,
      PnP, PnP-ADMM and MCMC all take the fast path — without touching a single solver.

    > It owns `estimate_anisotropy_ratio`. In v1 that computation lived in `matirf/gui/`
      and was referenced from `matirf/algorithms/`, which made the algorithm layer import
      the GUI layer — the circular import that forced lazy imports throughout v1. It is a
      property of the physics, so it belongs here, and the GUI now reads it from the
      operator instead of the reverse.
"""

import torch

import settings as settings
from core import Feature, ForwardOperator, features
from . import physics


class MatirfOperator(ForwardOperator):
    """
    Multi-angle TIRF: H[i, j] is the contribution of depth slice j to the stack measured at
    incidence angle i.

    f has shape (nz, ny, nx) and g has shape (n_angles, ny, nx); the operator acts only along
    the first axis, identically for every pixel, which is why a matrix product suffices.
    """

    name = "matirf"
    features = features(Feature.THREE_D, Feature.ANISOTROPIC, Feature.SCALE_AMBIGUOUS)

    def __init__(self, matrix: torch.Tensor, measurement_params: dict = None,
                 operator_params: dict = None):
        self.H = matrix
        self.measurement_params = measurement_params or {}
        self.operator_params = operator_params or {}
        self._normal_inverse_cache = {}

    @classmethod
    def from_config(cls, config: dict) -> "MatirfOperator":
        """Build H by reading the measurement .json named in the config."""
        from fileio import load_json

        return cls.from_measurement(load_json(config["input-paths"]["json"]),
                                    config.get("oper-params", {}))

    @classmethod
    def from_measurement(cls, measurement: dict, oper: dict) -> "MatirfOperator":
        """
        Build H from an already-loaded measurement parameter dict.

        This is the form the pipeline uses, because preprocessing may have removed
        background stacks and their angles: H must be built from the surviving angles, not
        from what the .json originally said.

        `normalize` is read defensively — its checkbox may never have been touched, so the
        key can be absent or stored as the string "null" right after a cache reset.
        """
        normalize = oper.get("normalize", False)
        if normalize in (None, "None", "null"):
            normalize = False
        matrix = physics.build_operator_matrix(
            angles_deg=measurement["angles_deg"],
            nz=oper["nz"], z0=oper["z0"], zN=oper["zN"],
            n_glass=measurement["n_glass"], n_medium=measurement["n_medium"],
            numerical_aperture=measurement["numerical_aperture"], n_oil=measurement["n_oil"],
            wavelength_nm=measurement["wavelength_nm"],
            beam_divergence_deg=measurement["beam_divergence_deg"],
            normalize=normalize,
        )
        return cls(matrix, measurement, oper)

    # ── the physics ──────────────────────────────────────────────────────────

    def apply(self, f: torch.Tensor) -> torch.Tensor:
        """H f — mix the nz depth slices into n_angles measured stacks, pixel by pixel."""
        return (self.H @ f.reshape(f.shape[0], -1)).reshape(self.H.shape[0], *f.shape[1:])

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """H^T y — back-project the stacks into depth slices."""
        Ht = self.H.transpose(0, 1)
        return (Ht @ y.reshape(y.shape[0], -1)).reshape(Ht.shape[0], *y.shape[1:])

    # ── the closed form that speeds up every proximal solver ─────────────────

    def solve_normal(self, b: torch.Tensor, lam: float = 0.0, **kwargs) -> torch.Tensor:
        """
        (H^T H + lam I)^-1 b, solved directly.

        H is at most a few dozen by a few dozen, so forming and inverting the normal matrix
        costs almost nothing and is exact, where the base class's conjugate gradient would
        only approximate it. Inverses are cached per lam because the solvers call this with
        the same lam every iteration — ADMM with a fixed mu, PnP-ADMM with a fixed rho.
        """
        inverse = self._normal_inverse_cache.get(lam)
        if inverse is None:
            n = self.H.shape[1]
            identity = torch.eye(n, dtype=settings.dtype, device=settings.device)
            inverse = torch.inverse(self.H.transpose(0, 1) @ self.H + lam * identity)
            self._normal_inverse_cache[lam] = inverse
        return (inverse @ b.reshape(b.shape[0], -1)).reshape(inverse.shape[0], *b.shape[1:])

    # ── a capability of the physics, which the GUI reads ─────────────────────

    def estimate_anisotropy_ratio(self) -> float:
        """
        delta = dz / dxy for this configuration — what the GUI's "Estimate" button reports.

        Living on the operator is what lets the algorithm layer stay free of any GUI import.
        """
        return physics.estimate_anisotropy_ratio(
            nz=self.operator_params["nz"],
            z0=self.operator_params["z0"],
            zN=self.operator_params["zN"],
            n_medium=self.measurement_params["n_medium"],
            numerical_aperture=self.measurement_params["numerical_aperture"],
            wavelength_nm=self.measurement_params["wavelength_nm"],
        )

    def __repr__(self) -> str:
        n_angles, nz = self.H.shape
        return f"<MatirfOperator {n_angles} angles x {nz} depth slices>"
