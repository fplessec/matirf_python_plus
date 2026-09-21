"""
REUSABLE GUI LAYER (Prebuilt UI components)

This module provides fully implemented, ready-to-use GUI components
for common functionalities across inverse problem applications
(matirf, deconv, and future extensions).

These components are built on top of the BASE layer and assemble
low-level parameter widgets into higher-level functional sections.

---------------------------------------------------------------------
Core idea
---------------------------------------------------------------------

While the BASE layer provides generic building blocks
(SimpleParameterWidget, BaseSectionWidget, BaseSectionQGroup),
the REUSABLE layer defines complete, domain-independent UI sections
that can be used directly without modification.

---------------------------------------------------------------------
What belongs here
---------------------------------------------------------------------

This layer includes prebuilt GUI modules such as:

- AddNoiseSection
    The simulated Poisson-Gaussian noise (photons N, read noise sigma), with its formula

- NoiseModelSection
    The noise model the reconstruction assumes: data fidelity and noise level (a, b)

- AlgorithmSelectionSection
    Full algorithm selector with:
        * algorithm registry integration
        * parameter panels per algorithm
        * dynamic switching of parameter sets
        * reset and persistence logic

- MessageSection
    Simple logging / message display panel for UI feedback

---------------------------------------------------------------------
Design role
---------------------------------------------------------------------

Reusable components:
    - compose BaseSectionWidget / BaseSectionQGroup
    - integrate application-level GUI logic (selection, state handling)
    - remain independent of specific inverse problem implementations
    - are safe to reuse across matirf, deconv, and future modules

---------------------------------------------------------------------
Key distinction
---------------------------------------------------------------------

- base/:
    low-level UI primitives (parameter rendering, layout, sync)

- reusable/:
    high-level prebuilt GUI sections (feature-complete building blocks)

- specializable/:
    abstract or extensible components requiring problem-specific override

- widgets/:
    low-level visual enhancements and presentation utilities
"""


# reusable components are grouped by kind (see the three sub-packages):
from .sections import (
    AddNoiseSection, ADD_NOISE_PARAMETERS_UI, add_noise_formula, AlgorithmSelectionSection,
    NoiseModelSection, NOISE_MODEL_UI, noise_model_formula,
    with_extra_buttons, MessageSection,
)
from .inputs import FileSelector, SelectorButton, JsonParametersEditor, PreprocessViewer
from .dialogs import FrequencyCutoffDialog, SingularValuePickerDialog