"""
GUI ARCHITECTURE OVERVIEW (Inverse Problems Framework)

This package implements the full GUI stack used by all inverse problem
modules (matirf, deconv, and future extensions such as super-resolution).

The system is organized into four layers, from low-level UI primitives
to full application skeletons.

=====================================================================
1. base/
=====================================================================

Low-level building blocks for parameter UIs.

This layer defines the core widgets used everywhere in the framework:
    - SimpleParameterWidget
    - BaseSectionWidget
    - BaseSectionQGroup

It is responsible for:
    - constructing parameter UIs from dictionaries
    - handling TOML synchronization
    - providing minimal reusable UI behavior

This layer is intentionally independent of any inverse problem logic.
It is the foundation everything else builds on.

=====================================================================
2. specializable/
=====================================================================

Application skeletons with extension points.

This layer defines full GUI structures (control windows, display windows)
that are meant to be subclassed by specific inverse problems.

It handles the heavy lifting:
    - window layout and menus
    - pipeline lifecycle integration
    - config loading/saving flow
    - signal wiring between UI and backend

What it does not define:
    - problem-specific parameters
    - algorithm details
    - visualization logic

Instead, it exposes hooks where subclasses plug in their behavior.

Think of it as a working application shell that becomes concrete
only when specialized.

=====================================================================
3. reusable/
=====================================================================

Prebuilt GUI components at the feature level.

This layer contains ready-to-use UI modules that can be dropped into
any application without modification.

Typical examples include:
    - noise configuration sections
    - algorithm selection + parameter panels
    - logging / message panels
    - standard composite UI blocks

These components are built on top of `base/` and are designed to be
independent of any specific inverse problem.

They help avoid reimplementing the same UI patterns across modules.

=====================================================================
4. widgets/
=====================================================================

UI utility components focused on presentation.

This layer contains visual building blocks that improve usability and
appearance but do not implement application logic.

Examples:
    - custom buttons and separators
    - text editors and display widgets
    - small interaction helpers

These widgets are used throughout the other layers to keep the UI
consistent and more pleasant to use.

=====================================================================

Design summary:

    base          → UI primitives
    specializable → application skeletons with hooks
    reusable      → feature-level UI modules
    widgets       → visual and interaction components

This structure makes it easy to:
    - add new inverse problems without rewriting the GUI
    - reuse full UI sections across projects
    - keep a consistent architecture across modules
    - avoid duplication of UI logic
"""