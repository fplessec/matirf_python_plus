"""
Object registry — the synthetic analogue of matirf.algorithms.ALGORITHMS.

OBJECT_TYPES maps a TOML section key -> SyntheticObjectType class. The ground-truth
TOML has one '[toml_key]' section per type (each with a 'count' and the type's
characteristic parameters), and the generator iterates this registry in order to build
the scene. Registration order is fixed, which keeps generation reproducible.
"""

from .base import ContinuousObject, SyntheticObjectType
from .ellipsoid import Ellipsoid, GaussianEllipsoid
from .filament import Filament3D, Filament
from .membrane import Membrane, MembraneSheet
from .double_layer import DoubleLayerType, DoubleLayer


# order matters: it fixes the RNG consumption order -> reproducible scenes
OBJECT_TYPES = {cls.toml_key: cls for cls in [
    Ellipsoid,
    Filament3D,
    Membrane,
    DoubleLayerType,
]}

__all__ = [
    "OBJECT_TYPES",
    "ContinuousObject",
    "SyntheticObjectType",
    "Ellipsoid", "GaussianEllipsoid",
    "Filament3D", "Filament",
    "Membrane", "MembraneSheet",
    "DoubleLayerType", "DoubleLayer",
]
