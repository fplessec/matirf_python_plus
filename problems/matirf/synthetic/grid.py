"""
The sampling grid — how the continuous scene becomes voxels.

MA-TIRF geometry: the volume is a very flat slab. Depth cannot usefully exceed a few hundred
nm (500-800 nm at most, set by the objective's maximum angle, and the deeper the harder to
reconstruct — 300 nm is where reconstructions are reliably good), while x and y span a
classical TIRF field of view. Hence an anisotropic grid: a fine axial step, a coarse lateral
one. Axis convention everywhere in the project: a volume is (Z, Y, X).

--------------------------------------------------------------------------------------
z_oversampling: a truth finer than the reconstruction
--------------------------------------------------------------------------------------

`nz` is the number of planes the RECONSTRUCTION will have. The truth itself is sampled on
`nz * z_oversampling` planes. Simulating the measurement from that finer truth, then
reconstructing on `nz` planes, avoids the "inverse crime" — generating and inverting with
the very same discretization, which makes every method look better than it is. The
reconstruction is then compared with the truth averaged back to `nz` planes (exact, since a
voxel value is the mean of the density over the voxel). See problems/matirf/problem.py.

Measured, for honesty: MA-TIRF's H varies slowly in depth, so at 50 planes over 300 nm the
model error this removes is only ~1e-4 of g — far below any realistic noise. It grows on
coarser grids (~3e-3 to 8e-3 at 10 planes). The other gain is practical: one truth of
150 planes can be reconstructed on 50, 30, 25, 15 or 10 planes, each with an exact
reference — which is what a benchmark varying nz needs.
"""

from dataclasses import dataclass

import torch

import settings as settings

GRID_TOML_KEY = "grid"


def _value(title, latex, default, unit="", dtype=float):
    return {"title": title, "type": "value",
            "param_info": {"dtype": dtype, "unit": unit, "latex_name": latex, "default": default}}


GRID_UI = {
    "nx": _value("Voxels along x", "n_x", 128, dtype=int),
    "ny": _value("Voxels along y", "n_y", 128, dtype=int),
    "nz": _value("Planes of the reconstruction", "n_z", 50, dtype=int),
    "z_oversampling": _value("Truth planes per reconstruction plane", "k_z", 3, dtype=int),
    "dxy_nm": _value("Lateral voxel size", "\\Delta xy", 100.0, "nm"),
    "z0_nm": _value("Smallest depth", "z_0", 0.0, "nm"),
    "zN_nm": _value("Largest depth", "z_N", 300.0, "nm"),
    "fine_step_nm": _value("Integration step", "\\Delta_{int}", 10.0, "nm"),
}


@dataclass(frozen=True)
class Grid:
    """The voxel grid of the TRUTH: nz * z_oversampling planes between z0 and zN."""

    nx: int
    ny: int
    nz: int
    dxy_nm: float
    z0_nm: float
    zN_nm: float
    z_oversampling: int = 1

    @classmethod
    def from_params(cls, params: dict) -> "Grid":
        return cls(nx=int(params["nx"]), ny=int(params["ny"]), nz=int(params["nz"]),
                   dxy_nm=float(params["dxy_nm"]), z0_nm=float(params["z0_nm"]),
                   zN_nm=float(params["zN_nm"]),
                   z_oversampling=max(1, int(params.get("z_oversampling", 1) or 1)))

    @property
    def planes(self) -> int:
        """Planes of the truth."""
        return self.nz * self.z_oversampling

    @property
    def shape(self) -> tuple:
        return (self.planes, self.ny, self.nx)

    @property
    def dz_nm(self) -> float:
        return (self.zN_nm - self.z0_nm) / self.planes

    @property
    def Lx_nm(self) -> float:
        return self.nx * self.dxy_nm

    @property
    def Ly_nm(self) -> float:
        return self.ny * self.dxy_nm

    # ── sub-voxel sampling, for the integration ──────────────────────────────

    def subvoxel_factors(self, step_nm: float, cap: int = 8) -> tuple:
        """Sub-samples per voxel along (xy, z) to reach about `step_nm`, capped."""
        return (max(1, min(cap, round(self.dxy_nm / step_nm))),
                max(1, min(cap, round(self.dz_nm / step_nm))))

    def fine_lateral_axes(self, ss: int) -> tuple:
        """Sub-voxel centres (nm) along x and y."""
        kwargs = {"device": settings.device, "dtype": settings.dtype}
        return (self.dxy_nm * (torch.arange(self.nx * ss, **kwargs) + 0.5) / ss,
                self.dxy_nm * (torch.arange(self.ny * ss, **kwargs) + 0.5) / ss)

    def fine_axial_slab(self, k: int, ss: int) -> torch.Tensor:
        """Sub-voxel centre depths (nm) inside truth plane `k`."""
        offsets = self.dz_nm * (torch.arange(ss, device=settings.device, dtype=settings.dtype) + 0.5) / ss
        return self.z0_nm + k * self.dz_nm + offsets
