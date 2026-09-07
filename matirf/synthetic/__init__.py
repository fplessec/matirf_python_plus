"""
matirf.synthetic — reproducible, TOML-driven synthetic ground truths for MA-TIRF.

Two concerns, two TOML files (see 'config.py'):
    > ground_truth.toml : the continuous objects themselves — which types, how many, and
      their characteristic parameters (grid-independent, defined in nm).
    > grid.toml         : how that continuous truth is sampled / visualized on the flat
      anisotropic MA-TIRF slab.

Each object type (Ellipsoid, Filament, Membrane, DoubleLayer) is a SyntheticObjectType
with its own 'ui_params' dictionary — exactly like an algorithm — and is registered in
OBJECT_TYPES. The GUI ('matirf synth' / python -m matirf.synthetic) renders one QGroupBox
of SimpleParameterWidgets per section, edits the TOML, and previews the result.

Noise is NOT part of the ground truth: g = H·f_true and its noise are produced by the
reconstruction pipeline via the '[add-noise]' section of the MA-TIRF config.

Programmatic use::

    from matirf.synthetic import load_grid_config, load_gt_config, generate_ground_truth
    f_true = generate_ground_truth(load_grid_config(), load_gt_config())
"""

from .grid import Grid, GRID_UI, GRID_TOML_KEY
from .objects import (
    OBJECT_TYPES,
    ContinuousObject,
    SyntheticObjectType,
    Ellipsoid, GaussianEllipsoid,
    Filament3D, Filament,
    Membrane, MembraneSheet,
    DoubleLayerType, DoubleLayer,
)
from .config import (
    GRID_CONFIG_PATH, GT_CONFIG_PATH,
    DEFAULT_GRID_CONFIG, DEFAULT_GT_CONFIG,
    SAMPLING_UI, SAMPLING_TOML_KEY,
    load_grid_config, load_gt_config,
    update_grid_cache, update_gt_cache,
)
from .generator import (
    build_objects,
    integrate_objects,
    normalize_01,
    generate_ground_truth,
    generate_from_cache,
    generate_and_save,
)

__all__ = [
    "Grid", "GRID_UI", "GRID_TOML_KEY",
    "OBJECT_TYPES", "ContinuousObject", "SyntheticObjectType",
    "Ellipsoid", "GaussianEllipsoid",
    "Filament3D", "Filament",
    "Membrane", "MembraneSheet",
    "DoubleLayerType", "DoubleLayer",
    "GRID_CONFIG_PATH", "GT_CONFIG_PATH",
    "DEFAULT_GRID_CONFIG", "DEFAULT_GT_CONFIG",
    "SAMPLING_UI", "SAMPLING_TOML_KEY",
    "load_grid_config", "load_gt_config",
    "update_grid_cache", "update_gt_cache",
    "build_objects", "integrate_objects", "normalize_01",
    "generate_ground_truth", "generate_from_cache", "generate_and_save",
]
