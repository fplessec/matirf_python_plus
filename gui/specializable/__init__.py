"""
SPECIALIZABLE GUI LAYER (Extensible application framework)

This module defines abstract and partially implemented GUI components
that form the structural backbone of inverse problem applications
(matirf, deconv, and future extensions such as super-resolution).

Unlike the BASE layer (UI primitives) and the REUSABLE layer
(prebuilt GUI sections), this layer defines full application structures
with explicit extension points ("hooks") that must be implemented
by problem-specific modules.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------

The SPECIALIZABLE layer provides complete GUI skeletons that are
functional but not fully defined.

Each component:
    - implements the full UI structure and interaction flow
    - defines a set of required hooks for customization
    - enforces architectural consistency across problems
    - minimizes duplication of application-level logic

---------------------------------------------------------------------
Main components
---------------------------------------------------------------------

1. Control Window (BaseControlWindow)
------------------------------------------------
Declarative main window for configuring and running inverse problems.

Provides:
    - automatic GUI construction from section descriptors
    - TOML-based configuration system
    - pipeline and display window integration
    - menu bar, shortcuts, and execution workflow

Extension points:
    - problem-specific sections (left/right layout)
    - pipeline manager class
    - display window manager class
    - cleanup logic on close

2. Display Window (BaseDisplayWindow)
------------------------------------------------
Main visualization window for reconstruction results.

Provides:
    - pipeline execution bridge (Qt-safe signals)
    - figure display layout (reconstruction + optional synthetic truth)
    - saving/loading of results
    - interactive switching between views

Extension points:
    - figure rendering logic
    - synthetic truth visualization
    - pipeline-specific configuration

3. Input Files Section (BaseInputFilesSection)
------------------------------------------------
Abstract UI component for handling input data selection.

Provides:
    - real vs synthetic mode toggle
    - file selection UI (image + metadata)
    - TOML synchronization logic

Requires implementation of:
    - file selector widgets
    - config parsing logic
    - mode persistence

4. Synthetic Truth Section (BaseSyntheticTruthSection)
------------------------------------------------
Optional analysis panel for synthetic experiments.

Provides:
    - metrics display table
    - difference visualization trigger
    - integration with pipeline results

Requires implementation of:
    - visualization of reconstruction error / difference maps

5. Infrastructure utilities
------------------------------------------------
- PipelineQtBridge:
    thread-safe communication layer between pipeline and Qt UI

- DisplayWindowManager:
    singleton manager for display windows lifecycle

---------------------------------------------------------------------
Design philosophy
---------------------------------------------------------------------

This layer defines the "application architecture contract":

    base         → UI primitives
    reusable     → ready-to-use UI sections
    specializable → full application skeletons with hooks

Key principles:
    - enforce consistent GUI structure across all inverse problems
    - centralize application logic (pipeline, config, execution flow)
    - isolate problem-specific logic into small override hooks
    - reduce duplication of control/display window implementations

---------------------------------------------------------------------
Summary
---------------------------------------------------------------------

SPECIALIZABLE = fully functional GUI applications with extension points.

They define *how the application behaves*, but not *what the problem is*.
"""