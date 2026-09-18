"""
MA-TIRF input files section — fully declarative: two FileSlot specs (the .tif image and the
.json measurement parameters). BaseInputFilesSection builds the selectors, the preview
button and the parameters editor from these specs.
"""

from common.gui.specializable.control_window import BaseInputFilesSection, FileSlot, Preview, Editor
from common.gui.widgets import ImageAndHisto3DViewer
from matirf import MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG, MATIRF_MEASUREMENTS_DIR
from matirf.cache import update_cache
from .measurement_parameters_ui_dictionary import MEASUREMENT_PARAMETERS_UI


_PREVIEW_TITLES = {
    "real": ("g_raw = input file (MA-TIRF image stack, raw measurement)",
             "g = preprocessed g_raw + noise"),
    "synthetic": ("f_true = input file (3D object, synthetic truth)",
                  "g_synth = H_synth @ f_true (preprocessed + noise)"),
}


def _compute_preview(config, mode):
    # the problem previews itself, through the same code path a real run takes, so the
    # preview can never disagree with the reconstruction that follows.
    from problems.matirf import MATIRF
    return MATIRF.preview(config)


def _preview_errors(config):
    """Missing-parameter messages shown if the MA-TIRF preview cannot be computed."""
    add_noise, oper = config.get('add-noise', {}), config.get('oper-params', {})
    msgs = []
    if add_noise.get('add_noise', False) and add_noise.get('sigma', 'None') == 'None':
        msgs.append('Noise standard deviation is None, please define a value.')
    for key, label in (('nz', 'Number of cuts on z'), ('z0', 'Smallest depth'), ('zN', 'Largest depth')):
        if oper.get(key, 'None') == 'None':
            msgs.append(f'{label} is None, please define a value.')
    return msgs


class InputFilesSection(BaseInputFilesSection):
    CONFIG_PATH = MATIRF_CONFIG_PATH
    DEFAULT_CONFIG = DEFAULT_MATIRF_CONFIG
    UPDATE_CACHE_FN = staticmethod(update_cache)
    MEASUREMENTS_DIR = MATIRF_MEASUREMENTS_DIR
    REAL_MODE_TEXT = "Work with real MA-TIRF measurement"
    SYNTHETIC_MODE_TEXT = "Simulate measurement with synthetic truth"
    DEFAULT_MODE_REAL = True

    IMAGE_SLOT = FileSlot(
        toml_key="tif", noun="tif", dialog_filter="Image Files (*.tif *.tiff)",
        title_real="Path of the MA-TIRF image stack",
        title_synthetic="Path of the 3D object (synthetic truth)",
        preview=Preview(ImageAndHisto3DViewer, _compute_preview,
                        titles=_PREVIEW_TITLES, size=(700, 900), error_check=_preview_errors),
    )
    JSON_SLOT = FileSlot(
        toml_key="json", noun="json", dialog_filter="Parameters Files (*.json)",
        title_real="Path of the measurement parameters",
        title_synthetic="Path of the simulated parameters",
        require_keys=MEASUREMENT_PARAMETERS_UI,                          # auto .json validator
        editor=Editor(ui=MEASUREMENT_PARAMETERS_UI, title_noun="Measurement Parameters", width=700),
    )
