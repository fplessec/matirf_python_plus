"""
Deconvolution PSF-parameters editor.

The PSF parameters are all plain SimpleParameterWidgets, so this is just the generic
JsonParametersEditor configured with the deconv PSF UI dict and paths — no extra fields.
"""

from common.gui.reusable import JsonParametersEditor
from deconv import DECONV_MEASUREMENTS_DIR

from .psf_parameters_ui_dictionary import PSF_PARAMETERS_UI


class PsfParametersEditor(JsonParametersEditor):

    def __init__(self, parent):
        super().__init__(
            parent,
            params_ui_dict=PSF_PARAMETERS_UI,
            measurements_dir=DECONV_MEASUREMENTS_DIR,
            title_noun="PSF Parameters",
            width=600,
        )
