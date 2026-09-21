"""
From a scene to a ground truth: draw the objects, integrate them on the grid, save.

    generate(scene) -> f_true                  (planes, ny, nx) tensor in [0, 1]
    save_truth(f_true, path, scene)            the TIF, plus its record <name>.truth.json
    read_record(tif_path) -> dict | None       that record, if the TIF has one
    regenerate(tif_path) -> f_true             the truth again, from its record alone
    downsample_z(f, factor)                    a fine truth averaged to coarser planes

The steps of `generate`:
    1. one seeded generator draws every object, kind after kind in the fixed order of
       OBJECTS: the same scene always gives the same objects
    2. each voxel becomes the MEAN of the summed density over its volume (midpoint
       quadrature on sub-samples ~`fine_step_nm` apart). The truth is thereby an average,
       which is what makes averaging planes (downsample_z) exact, and it separates the
       continuous object from its discretization. An object is evaluated only inside its
       bounding box, and only on the planes its box reaches
    3. normalized to [0, 1]

No noise here: g = H f_true and its noise are the reconstruction pipeline's business
('[add-noise]'), so one truth serves every noise level.

The record (<name>.truth.json) makes a TIF self-describing: its geometry — which the
MA-TIRF problem checks and uses (the depth range it lives in, the number of planes to
reconstruct) — and the complete scene, from which it is reproduced bit for bit.
"""

import json
from pathlib import Path

import torch

import settings as settings
from fileio import save_tif
from .grid import Grid, GRID_TOML_KEY
from .objects import OBJECTS
from .scene import complete, SAMPLING_TOML_KEY

RECORD_FORMAT = "matirf-synthetic-truth/2"


def build_objects(scene: dict, grid: Grid) -> list:
    """Every object of the scene, drawn from its seed (reproducible)."""
    generator = torch.Generator().manual_seed(int(scene[SAMPLING_TOML_KEY]["seed"]))
    objects = []
    for key, cls in OBJECTS.items():              # fixed order: fixed RNG consumption
        objects += cls.build(scene.get(key, {}), generator, grid)
    return objects


def integrate(objects: list, grid: Grid, step_nm: float) -> torch.Tensor:
    """The voxel means of the summed densities, on the truth grid (planes, ny, nx)."""
    ss_xy, ss_z = grid.subvoxel_factors(step_nm)
    xf, yf = grid.fine_lateral_axes(ss_xy)

    ## each object's lateral window on the fine grid, built once for the whole volume
    windows = []
    for obj in objects:
        x0, x1, y0, y1, z0, z1 = obj.bounds()
        ix = slice(int(torch.searchsorted(xf, x0)), int(torch.searchsorted(xf, x1, right=True)))
        iy = slice(int(torch.searchsorted(yf, y0)), int(torch.searchsorted(yf, y1, right=True)))
        if ix.start < ix.stop and iy.start < iy.stop:
            Y, X = torch.meshgrid(yf[iy], xf[ix], indexing="ij")
            windows.append((obj, iy, ix, Y, X, z0, z1))

    f = torch.zeros(grid.shape, device=settings.device, dtype=settings.dtype)
    for k in range(grid.planes):
        depths = grid.fine_axial_slab(k, ss_z)
        low, high = grid.z0_nm + k * grid.dz_nm, grid.z0_nm + (k + 1) * grid.dz_nm
        plane = torch.zeros((ss_z, yf.numel(), xf.numel()), device=settings.device,
                            dtype=settings.dtype)
        for obj, iy, ix, Y, X, z0, z1 in windows:
            if z1 < low or z0 > high:
                continue
            for p, depth in enumerate(depths):
                plane[p, iy, ix] += obj.density(X, Y, torch.full_like(X, float(depth)))
        f[k] = plane.reshape(ss_z, grid.ny, ss_xy, grid.nx, ss_xy).mean(dim=(0, 2, 4))
    return f


def generate(scene: dict) -> torch.Tensor:
    """The ground truth of a scene, (planes, ny, nx) in [0, 1] — deterministic."""
    scene = complete(scene)
    grid = Grid.from_params(scene[GRID_TOML_KEY])
    f = integrate(build_objects(scene, grid), grid,
                  float(scene[GRID_TOML_KEY].get("fine_step_nm", 25.0)))
    peak = f.max()
    return f / peak if peak > 0 else f


# ── saving a truth with its record ────────────────────────────────────────────

def record_path(tif_path) -> Path:
    return Path(tif_path).with_suffix(".truth.json")


def geometry(scene: dict) -> dict:
    """What the MA-TIRF problem needs to know about a truth file."""
    grid = Grid.from_params(complete(scene)[GRID_TOML_KEY])
    return {"planes": grid.planes, "nz": grid.nz, "z_oversampling": grid.z_oversampling,
            "z0_nm": grid.z0_nm, "zN_nm": grid.zN_nm, "dxy_nm": grid.dxy_nm,
            "nx": grid.nx, "ny": grid.ny}


def save_truth(f_true: torch.Tensor, path, scene: dict) -> Path:
    """The TIF at `path` and its record next to it. Returns the TIF path."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        path = path.with_suffix(".TIF")
    save_tif(f_true, str(path))
    record = {"format": RECORD_FORMAT, "geometry": geometry(scene), "scene": complete(scene)}
    record_path(path).write_text(json.dumps(record, indent=2, default=str))
    return path


def read_record(tif_path):
    """The record saved with a truth TIF, or None for a TIF without one."""
    path = record_path(tif_path)
    return json.loads(path.read_text()) if path.exists() else None


def regenerate(tif_path) -> torch.Tensor:
    record = read_record(tif_path)
    if record is None or "scene" not in record:
        raise ValueError(f"{tif_path} has no scene record to regenerate it from")
    return generate(record["scene"])


def downsample_z(f: torch.Tensor, factor: int) -> torch.Tensor:
    """Average consecutive groups of `factor` planes — exact for voxel means."""
    if factor == 1:
        return f
    planes, ny, nx = f.shape
    if planes % factor:
        raise ValueError(f"{planes} planes cannot be grouped by {factor}")
    return f.reshape(planes // factor, factor, ny, nx).mean(dim=1)
