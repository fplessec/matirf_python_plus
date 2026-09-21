"""
The kinds of synthetic objects, and their registry.

    Ellipsoid   vesicles, endosomes, focal adhesions        (ellipsoid.py)
                (at random depths, or on a plane / sphere)
    Filament    actin stress fibres, microtubules            (filament.py)
    Membrane    the basal membrane of an adherent cell       (membrane.py)

OBJECTS maps a scene's TOML section key to its class. Its order is fixed on purpose: the
generator draws every object from ONE seeded generator in this order, so the order is part
of what makes a scene reproducible. Add a new kind at the END, never in the middle.

ADDING ONE: write a SyntheticObject subclass (see base.py) and append it below.
"""

from .base import SyntheticObject, value_param, count_param, gate_by_count
from .ellipsoid import Ellipsoid
from .filament import Filament
from .membrane import Membrane

OBJECTS = {cls.toml_key: cls for cls in (Ellipsoid, Filament, Membrane)}

__all__ = ["OBJECTS", "SyntheticObject", "Ellipsoid", "Filament", "Membrane",
           "value_param", "count_param", "gate_by_count"]
