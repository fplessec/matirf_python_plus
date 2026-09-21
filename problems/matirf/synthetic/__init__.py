"""
problems.matirf.synthetic — plausible, reproducible synthetic ground truths for MA-TIRF.

What is generated: a small piece of an adherent cell as MA-TIRF sees it — a flat slab a few
hundred nm deep (300 nm by default: where reconstructions are reliable) under a classical
TIRF field of view. Three kinds of objects, each one class (objects/):

    Ellipsoid   vesicles, endosomes, focal adhesions
    Filament    actin stress fibres, microtubules
    Membrane    the basal membrane of an adherent cell

A SCENE (scene.py) is one TOML file saying everything: the grid, the seed, and how many of
each object with which sizes. The generator (generator.py) turns it into a (Z, Y, X)
volume in [0, 1] and saves it as a TIF with a record, <name>.truth.json, holding its
geometry and its scene — so a truth can always be checked and reproduced.

Ready-made scenes (presets/) are the benchmark's truths: vesicles, fibres, cell, and
cell_fibres_vesicles (the three together).

The truth is sampled `z_oversampling` times finer in depth than the reconstruction will be
(grid.py): the measurement is simulated from the fine truth and the reconstruction is
compared with the truth averaged back — no inverse crime.

    matirf synth                       the generator window (gui.py)

    from problems.matirf.synthetic import load_preset, generate, save_truth
    scene = load_preset("cell")
    save_truth(generate(scene), "truths/cell.TIF", scene)

Noise is NOT part of the truth: it is added when the measurement is simulated, through the
'[add-noise]' section of the MA-TIRF config, so one truth serves every noise level.
"""

from .grid import Grid, GRID_UI, GRID_TOML_KEY
from .objects import OBJECTS, SyntheticObject, Ellipsoid, Filament, Membrane
from .scene import (
    SCENE_PATH, PRESETS_DIR, DEFAULT_SCENE, SAMPLING_UI, SAMPLING_TOML_KEY,
    load_scene, save_scene, presets, load_preset, update_scene_cache,
)
from .generator import (
    generate, save_truth, read_record, record_path, regenerate, downsample_z, geometry,
)

__all__ = [
    "Grid", "GRID_UI", "GRID_TOML_KEY",
    "OBJECTS", "SyntheticObject", "Ellipsoid", "Filament", "Membrane",
    "SCENE_PATH", "PRESETS_DIR", "DEFAULT_SCENE", "SAMPLING_UI", "SAMPLING_TOML_KEY",
    "load_scene", "save_scene", "presets", "load_preset", "update_scene_cache",
    "generate", "save_truth", "read_record", "record_path", "regenerate", "downsample_z",
    "geometry",
]
