"""
Generation: two TOML configs -> continuous objects -> anisotropic voxels -> f_true.

    1. Build the continuous objects from the ground-truth config: for each object type
       in the registry, its '[section]' provides a 'count' and characteristic params; a
       single seeded torch.Generator draws every instance (reproducible).
    2. INTEGRATE the summed density onto the anisotropic grid by midpoint quadrature —
       each voxel is the average over 'ssxy × ssxy × ssz' sub-samples of its physical
       extent. This separates the true object from its discretization.
    3. Normalize to [0, 1].

Noise is NOT added here: g = H·f_true and its noise are produced by the reconstruction
pipeline via the '[add-noise]' section of the MA-TIRF config.

Output: a (nz, ny, nx) torch tensor in [0, 1], ready to be saved as a TIF and used by
the existing synthetic mode.
"""

import torch

import common.settings as settings
from common.in_out import save_tif
from .grid import Grid
from .objects import OBJECT_TYPES
from .config import (
    load_grid_config, load_gt_config, GRID_TOML_KEY, SAMPLING_TOML_KEY,
)


def build_objects(gt_config: dict, grid: Grid, seed: int) -> list:
    """Instantiates every object of every type from the ground-truth config (reproducible)."""
    gen = torch.Generator().manual_seed(int(seed))
    objects = []
    for key, cls in OBJECT_TYPES.items():  # fixed order -> reproducible RNG consumption
        params = gt_config.get(key, {})
        objects += cls.build(params, gen, grid)
    return objects


def integrate_objects(objects: list, grid: Grid,
                      fine_step_nm: float = 25.0, supersample_cap: int = 8) -> torch.Tensor:
    """
    Integrates a list of continuous objects onto the anisotropic 'grid' by midpoint
    quadrature, returning a (nz, ny, nx) tensor. Memory is bounded by processing one
    axial voxel at a time.
    """
    ssxy, ssz = grid.subvoxel_factors(fine_step_nm, cap=supersample_cap)
    xf, yf = grid.fine_lateral_axes(ssxy)
    Yf, Xf = torch.meshgrid(yf, xf, indexing='ij')  # (ny·ssxy, nx·ssxy)

    f = torch.zeros(grid.shape, device=settings.device, dtype=settings.dtype)
    for k in range(grid.nz):
        z_subs = grid.fine_axial_slab(k, ssz)
        acc = torch.zeros((ssz, *Xf.shape), device=settings.device, dtype=settings.dtype)
        for p, zc in enumerate(z_subs):
            Z = torch.full_like(Xf, float(zc))
            plane = torch.zeros_like(Xf)
            for obj in objects:
                plane = plane + obj.density(Xf, Yf, Z)
            acc[p] = plane
        f[k] = acc.reshape(ssz, grid.ny, ssxy, grid.nx, ssxy).mean(dim=(0, 2, 4))
    return f


def normalize_01(f: torch.Tensor) -> torch.Tensor:
    """Normalizes a non-negative tensor to [0, 1] (zeros if it is all zero)."""
    fmax = torch.max(f)
    if fmax <= 0:
        return torch.zeros_like(f)
    return f / fmax


def generate_ground_truth(grid_config: dict, gt_config: dict) -> torch.Tensor:
    """
    Full generation from the two configs -> (nz, ny, nx) tensor in [0, 1], deterministic
    given the seed in gt_config['sampling']['seed'].
    """
    grid_params = grid_config[GRID_TOML_KEY]
    grid = Grid.from_params(grid_params)
    seed = gt_config.get(SAMPLING_TOML_KEY, {}).get("seed", 0)
    fine_step = float(grid_params.get("fine_step_nm", 25.0))
    objects = build_objects(gt_config, grid, seed)
    f = integrate_objects(objects, grid, fine_step_nm=fine_step)
    return normalize_01(f)


def generate_from_cache() -> torch.Tensor:
    """Generates from the two cached TOML files (grid.toml + ground_truth.toml)."""
    return generate_ground_truth(load_grid_config(), load_gt_config())


def generate_and_save(grid_config: dict, gt_config: dict, filepath) -> torch.Tensor:
    """Generates and saves the ground truth as a TIF (ZYX). Returns it."""
    f_true = generate_ground_truth(grid_config, gt_config)
    save_tif(f_true, filepath)
    return f_true
