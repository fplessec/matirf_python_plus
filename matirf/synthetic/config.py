"""
The two TOML files that drive the package, and their default contents.

    grid.toml          — the '[grid]' section: how the continuous truth is sampled and
                         visualized (nx, ny, nz, dxy, z0, zN, fine step). One QGroupBox.

    ground_truth.toml  — the continuous objects themselves, fully: a '[sampling]' section
                         (seed) plus one '[<object>]' section per object type, each with a
                         'count' and the type's characteristic parameters. One QGroupBox
                         per section.

Defaults are derived directly from the UI dictionaries (GRID_UI, SAMPLING_UI, and each
object type's ui_params), so there is a single source of truth for parameters.
"""

from pathlib import Path

from common.in_out import load_or_create_toml, save_toml
from common.cache import make_update_cache
from .grid import GRID_UI, GRID_TOML_KEY
from .objects import OBJECT_TYPES
from .objects.base import _default_of, value_param


SYNTHETIC_DIR = Path(__file__).resolve().parent
CACHE_DIR = SYNTHETIC_DIR / "cache"
GRID_CONFIG_PATH = CACHE_DIR / "grid.toml"
GT_CONFIG_PATH = CACHE_DIR / "ground_truth.toml"

# the '[sampling]' section of the ground-truth TOML (just the master seed for now)
SAMPLING_TOML_KEY = "sampling"
SAMPLING_UI = {
    "seed": value_param("Random seed", "\\text{seed}", 42, dtype=int),
}


def _defaults_from_ui(ui: dict) -> dict:
    return {name: _default_of(cfg) for name, cfg in ui.items()}


def _section_count(section) -> int:
    try:
        return int(section.get("count", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def normalize_gt(config: dict) -> bool:
    """
    Enforces the invariant on the ground-truth config, IN PLACE:
        > every object '[section]' has a 'count' ;
        > when count == 0 the section is pruned to just '{count = 0}'
          (a disabled object carries no characteristic parameters) ;
        > when count > 0 the section is left untouched (missing parameters fall back to
          the type's defaults at generation time — see SyntheticObjectType.build).

    Returns True if anything changed. This is what makes a count-0 object collapse in the
    TOML, mirroring the UI where its parameters are hidden.
    """
    changed = False
    for key, cls in OBJECT_TYPES.items():
        section = config.get(key)
        if not isinstance(section, dict):
            config[key] = {"count": cls.default_params()["count"]}
            changed = True
            continue
        if "count" not in section:
            section["count"] = cls.default_params()["count"]
            changed = True
        if _section_count(section) <= 0 and set(section.keys()) != {"count"}:
            config[key] = {"count": _section_count(section)}
            changed = True
    return changed


# object sections default to just {count}; only count>0 types keep their full parameters
DEFAULT_GRID_CONFIG = {GRID_TOML_KEY: _defaults_from_ui(GRID_UI)}

DEFAULT_GT_CONFIG = {
    SAMPLING_TOML_KEY: _defaults_from_ui(SAMPLING_UI),
    **{key: cls.default_params() for key, cls in OBJECT_TYPES.items()},
}
normalize_gt(DEFAULT_GT_CONFIG)


def load_grid_config(path=GRID_CONFIG_PATH) -> dict:
    """Loads grid.toml, filling any missing key from the defaults (grid has no gating)."""
    config = load_or_create_toml(path, DEFAULT_GRID_CONFIG)
    changed = False
    grid = config.setdefault(GRID_TOML_KEY, {})
    for key, value in DEFAULT_GRID_CONFIG[GRID_TOML_KEY].items():
        if key not in grid:
            grid[key] = value
            changed = True
    if changed:
        save_toml(config, path)
    return config


def load_gt_config(path=GT_CONFIG_PATH) -> dict:
    """Loads ground_truth.toml, ensuring the seed and applying the count-gating invariant."""
    config = load_or_create_toml(path, DEFAULT_GT_CONFIG)
    changed = False
    sampling = config.setdefault(SAMPLING_TOML_KEY, {})
    for key, value in _defaults_from_ui(SAMPLING_UI).items():
        if key not in sampling:
            sampling[key] = value
            changed = True
    changed = normalize_gt(config) or changed
    if changed:
        save_toml(config, path)
    return config


# cache writers bound to each file (key_path, value) -> writes into the TOML
update_grid_cache = make_update_cache(GRID_CONFIG_PATH, DEFAULT_GRID_CONFIG)
update_gt_cache = make_update_cache(GT_CONFIG_PATH, DEFAULT_GT_CONFIG)
