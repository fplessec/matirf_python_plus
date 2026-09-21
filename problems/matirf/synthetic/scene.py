"""
A scene — ONE TOML file describing a synthetic ground truth completely.

    [grid]        how it is sampled: nx, ny, nz, z_oversampling, dxy_nm, z0_nm, zN_nm, ...
    [sampling]    the seed every object is drawn from
    [ellipsoid]   count + parameters        (one section per kind of object, see objects/)
    [filament]    count + parameters
    [membrane]    count + parameters

A kind whose count is 0 keeps only `count = 0`, mirroring the interface, where its
parameters are hidden. A parameter missing from a section falls back to its default.

Where scenes live:
    cache/scene.toml   the one the generator window edits (not versioned)
    presets/*.toml     ready-made, biologically meaningful scenes — the benchmark's truths
"""

from pathlib import Path

from fileio import load_or_create_toml, save_toml
from fileio.cache import make_update_cache
from .grid import GRID_UI, GRID_TOML_KEY
from .objects import OBJECTS
from .objects.base import _default_of, value_param

SYNTHETIC_DIR = Path(__file__).resolve().parent
SCENE_PATH = SYNTHETIC_DIR / "cache" / "scene.toml"
PRESETS_DIR = SYNTHETIC_DIR / "presets"

SAMPLING_TOML_KEY = "sampling"
SAMPLING_UI = {"seed": value_param("Random seed", "\\text{seed}", 0, dtype=int)}


def _defaults(ui: dict) -> dict:
    return {name: _default_of(config) for name, config in ui.items()}


def prune(scene: dict) -> dict:
    """The scene with every kind present, and a count-0 kind reduced to {count: 0}."""
    scene = dict(scene)
    for key, cls in OBJECTS.items():
        section = dict(scene.get(key) or {"count": cls.default_params()["count"]})
        count = int(section.get("count", 0) or 0)
        scene[key] = section if count > 0 else {"count": 0}
    return scene


DEFAULT_SCENE = prune({
    GRID_TOML_KEY: _defaults(GRID_UI),
    SAMPLING_TOML_KEY: _defaults(SAMPLING_UI),
    **{key: cls.default_params() for key, cls in OBJECTS.items()},
})


def complete(scene: dict) -> dict:
    """Missing grid and sampling values filled from the defaults; objects pruned."""
    scene = dict(scene)
    scene[GRID_TOML_KEY] = {**DEFAULT_SCENE[GRID_TOML_KEY], **scene.get(GRID_TOML_KEY, {})}
    scene[SAMPLING_TOML_KEY] = {**DEFAULT_SCENE[SAMPLING_TOML_KEY],
                                **scene.get(SAMPLING_TOML_KEY, {})}
    return prune(scene)


def load_scene(path=SCENE_PATH) -> dict:
    """A scene file, completed (and, for the cache, rewritten if it was incomplete)."""
    raw = load_or_create_toml(path, DEFAULT_SCENE)
    scene = complete(raw)
    if scene != raw and Path(path) == SCENE_PATH:
        save_toml(scene, path)
    return scene


def save_scene(scene: dict, path) -> None:
    save_toml(complete(scene), path)


def presets() -> list:
    """The ready-made scenes, by name."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.toml"))


def load_preset(name: str) -> dict:
    path = PRESETS_DIR / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no preset {name!r} (available: {', '.join(presets())})")
    return load_scene(path)


## writes one value into the cached scene: (key_path, value)
update_scene_cache = make_update_cache(SCENE_PATH, DEFAULT_SCENE)
