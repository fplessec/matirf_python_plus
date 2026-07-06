"""
BASE GUI LAYER (Primitive parameter UI system)

This module defines the lowest-level building blocks used to construct
parameter-based user interfaces for inverse problems.

It implements a 3-level hierarchical abstraction:

---------------------------------------------------------------------
1. SimpleParameterWidget
---------------------------------------------------------------------
A single-line widget representing one parameter.

It is responsible for:
    - displaying a parameter (value / boolean / option)
    - handling user input
    - synchronizing with a TOML-backed configuration
    - optionally exposing an "extra action" button

This is the atomic UI unit of the system.

---------------------------------------------------------------------
2. BaseSectionWidget
---------------------------------------------------------------------
A vertical composition of SimpleParameterWidget instances.

It is responsible for:
    - building a full parameter section from a UI dictionary
    - stacking multiple parameters with separators
    - exposing a unified interface to access / reset / sync values

This is the core reusable section builder.

---------------------------------------------------------------------
3. BaseSectionQGroup
---------------------------------------------------------------------
A visual wrapper around BaseSectionWidget using a QGroupBox.

It is responsible for:
    - grouping parameters under a titled frame
    - providing a collapsible / visually structured section
    - integrating BaseSectionWidget into higher-level layouts

---------------------------------------------------------------------

Overall design philosophy:
    - UI can be fully constructed from dictionaries
    - each layer composes the previous one
    - separation between:
        * parameter logic (SimpleParameterWidget)
        * section composition (BaseSectionWidget)
        * visual grouping (BaseSectionQGroup)

This module is framework-agnostic (i.e. independent) with respect to
inverse problems: it only defines generic parameter UI primitives.
"""


from .single_parameter_widget import SimpleParameterWidget
from .base_section_widget import BaseSectionWidget
from .base_section_qgroup import BaseSectionQGroup