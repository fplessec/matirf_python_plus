"""
Anisotropic sampling grid — the "how it is discretized / visualized" half.

The ground truth is defined as continuous objects in nm (grid-independent, see the
'objects' subpackage). The Grid is the separate concern that says how those objects
are SAMPLED onto voxels: it is driven by its own TOML section '[grid]' (rendered in
the GUI via GRID_UI), exactly like the operator parameters of the control window.

MA-TIRF geometry: the volume is a very flat slab. Depth (z) cannot exceed ~500 nm,
whereas x and y span tens of micrometres. The grid is therefore anisotropic — a fine
axial step 'dz' (super-resolved) and a coarse lateral step 'dxy'.

Axis convention (as everywhere in the project): a 3D image is (Z, Y, X), so the
ground truth tensor has shape (nz, ny, nx).
"""

from dataclasses import dataclass

import torch

import common.settings as settings


# TOML section key + parameter UI dictionary for the grid (used by the GUI and to
# build the default config). Same format as OPERATOR_PARAMETERS_UI in the control window.
GRID_TOML_KEY = "grid"

GRID_UI = {
    "nx": {"title": "Voxels along x", "type": "value",
           "param_info": {"dtype": int, "unit": "", "latex_name": "n_x", "default": 64}},
    "ny": {"title": "Voxels along y", "type": "value",
           "param_info": {"dtype": int, "unit": "", "latex_name": "n_y", "default": 64}},
    "nz": {"title": "Voxels along z (axial cuts)", "type": "value",
           "param_info": {"dtype": int, "unit": "", "latex_name": "n_z", "default": 30}},
    "dxy_nm": {"title": "Lateral voxel size", "type": "value",
               "param_info": {"dtype": float, "unit": "nm", "latex_name": "\\Delta xy", "default": 100.0}},
    "z0_nm": {"title": "Smallest depth", "type": "value",
              "param_info": {"dtype": float, "unit": "nm", "latex_name": "z_0", "default": 0.0}},
    "zN_nm": {"title": "Largest depth", "type": "value",
              "param_info": {"dtype": float, "unit": "nm", "latex_name": "z_N", "default": 300.0}},
    "fine_step_nm": {"title": "Integration fine step", "type": "value",
                     "param_info": {"dtype": float, "unit": "nm", "latex_name": "\\Delta_{fine}", "default": 25.0}},
}


@dataclass(frozen=True)
class Grid:
    """
    Anisotropic voxel grid on which the continuous ground truth is integrated.

    Frozen because a grid is a value object: a fixed coordinate system. Freezing makes
    it hashable and prevents accidental mutation while objects are being sampled onto it
    (all derived quantities assume nz / z0 / zN stay consistent).
    """

    nx: int
    ny: int
    nz: int
    dxy_nm: float
    z0_nm: float
    zN_nm: float

    # ── derived quantities ────────────────────────────────────────────────

    @property
    def dz_nm(self) -> float:
        """Axial voxel size Δz = (zN - z0) / nz, in nanometers."""
        return (self.zN_nm - self.z0_nm) / self.nz

    @property
    def Lx_nm(self) -> float:
        """Lateral extent along x, in nanometers."""
        return self.nx * self.dxy_nm

    @property
    def Ly_nm(self) -> float:
        """Lateral extent along y, in nanometers."""
        return self.ny * self.dxy_nm

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape (nz, ny, nx) of the ground truth tensor."""
        return (self.nz, self.ny, self.nx)

    @property
    def anisotropy_ratio(self) -> float:
        """Ratio δ = Δz / Δxy — how much finer the axial sampling is than lateral."""
        return self.dz_nm / self.dxy_nm

    # ── super-sampling for integration ────────────────────────────────────

    def subvoxel_factors(self, fine_step_nm: float, cap: int = 8) -> tuple[int, int]:
        """Number of sub-samples per voxel along (xy, z) to reach ~'fine_step_nm', capped."""
        ssxy = max(1, min(cap, round(self.dxy_nm / fine_step_nm)))
        ssz = max(1, min(cap, round(self.dz_nm / fine_step_nm)))
        return ssxy, ssz

    def fine_lateral_axes(self, ssxy: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sub-voxel centre coordinates (nm) along x and y for lateral super-sampling 'ssxy'."""
        xf = self.dxy_nm * (torch.arange(self.nx * ssxy, device=settings.device,
                                         dtype=settings.dtype) + 0.5) / ssxy
        yf = self.dxy_nm * (torch.arange(self.ny * ssxy, device=settings.device,
                                         dtype=settings.dtype) + 0.5) / ssxy
        return xf, yf

    def fine_axial_slab(self, k: int, ssz: int) -> torch.Tensor:
        """Sub-voxel centre depths (nm) inside the axial voxel of index 'k'."""
        base = self.z0_nm + k * self.dz_nm
        offsets = self.dz_nm * (torch.arange(ssz, device=settings.device,
                                             dtype=settings.dtype) + 0.5) / ssz
        return base + offsets

    # ── config bridge ─────────────────────────────────────────────────────

    @classmethod
    def from_params(cls, params: dict) -> "Grid":
        """Builds a Grid from a '[grid]' TOML section (ignores extra keys like fine_step_nm)."""
        return cls(
            nx=int(params["nx"]), ny=int(params["ny"]), nz=int(params["nz"]),
            dxy_nm=float(params["dxy_nm"]),
            z0_nm=float(params["z0_nm"]), zN_nm=float(params["zN_nm"]),
        )
