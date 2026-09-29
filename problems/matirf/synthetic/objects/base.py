"""
SyntheticObject — the one class every kind of synthetic object derives from.

A kind of object (ellipsoid, filament, membrane) is ONE class, playing two roles:

    the class       describes the kind: its `name`, its TOML section `toml_key`, its
                    parameters `ui_params` (the same UI format as an algorithm's), and how
                    to draw one instance at random: `sample(params, gen, grid)`
    an instance     is one object in the scene: it knows where it is (`bounds`) and how
                    much fluorophore it holds at any point (`density`)

    class Vesicle(SyntheticObject):
        name, toml_key = "Vesicle", "vesicle"
        ui_params = {"count": count_param("vesicles"), "radius_nm": value_param(...)}

        @classmethod
        def sample(cls, params, gen, grid):
            return cls(...)                      # one instance, drawn with `gen`

        def bounds(self):                        # (x0, x1, y0, y1, z0, z1) in nm
            ...
        def density(self, X, Y, Z):              # non-negative, same shape as X
            ...

then one line in objects/__init__.py (OBJECTS) — the interface, the TOML and the generator
pick it up with no other change.

Conventions: every length is in nm; a density is non-negative (a fluorophore
concentration: the reconstruction assumes f >= 0); randomness comes ONLY from `gen`, the
scene's seeded generator, so a scene is reproducible bit for bit.
"""

import math


class SyntheticObject:
    """Base of every kind of synthetic object (see the module docstring)."""

    name = ""
    toml_key = ""
    ui_params = {}

    # ── the kind ─────────────────────────────────────────────────────────────

    @classmethod
    def default_params(cls) -> dict:
        return {name: _default_of(config) for name, config in cls.ui_params.items()}

    @classmethod
    def build(cls, params: dict, gen, grid) -> list:
        """`count` instances; a missing parameter falls back to its default."""
        full = {**cls.default_params(), **(params or {})}
        count = int(full.get("count", 0) or 0)
        return [cls.sample(full, gen, grid) for _ in range(max(0, count))]

    @classmethod
    def sample(cls, params: dict, gen, grid) -> "SyntheticObject":
        raise NotImplementedError

    # ── one instance ─────────────────────────────────────────────────────────

    def bounds(self) -> tuple:
        """(x0, x1, y0, y1, z0, z1) in nm, outside which the density is negligible.

        Used to evaluate the density only where it matters — a filament covers a few
        percent of the field. The default, everywhere, is always correct, never fast."""
        return (-math.inf, math.inf, -math.inf, math.inf, -math.inf, math.inf)

    def density(self, X, Y, Z):
        raise NotImplementedError


## how far from its core an object's density is still evaluated, in standard deviations:
## exp(-0.5 * 6^2) ~ 1.5e-8, below float32 resolution relative to the peak
REACH = 6.0


# ── declaring parameters concisely ────────────────────────────────────────────

def value_param(title, latex_name, default, unit="", dtype=float):
    """A numeric parameter (number field + latex display)."""
    return {"title": title, "type": "value",
            "param_info": {"dtype": dtype, "unit": unit,
                           "latex_name": latex_name, "default": default}}


def count_param(noun_plural, default=0):
    """The mandatory `count` parameter: how many instances of this kind in the scene."""
    return value_param(f"Number of {noun_plural}", "N", default, unit="", dtype=int)


def _default_of(config: dict):
    if config["type"] == "option":
        return config["param_info"]["options_list"][0]
    return config["param_info"]["default"]


def gate_by_count(ui_params: dict) -> dict:
    """A copy where every parameter but `count` is shown only when count > 0."""
    def positive(value):
        try:
            return (value or 0) > 0
        except TypeError:
            return False
    return {name: config if name == "count"
            else {**config, "depends_on": {**config.get("depends_on", {}), "count": positive}}
            for name, config in ui_params.items()}
