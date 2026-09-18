"""
Deconvolution input files section — fully declarative: two FileSlot specs (the .png image
and the .json PSF parameters). BaseInputFilesSection builds the selectors, the preview
button and the (generic) PSF-parameters editor from these specs.
"""

from common.gui.specializable.control_window import BaseInputFilesSection, FileSlot, Preview, Editor
from common.gui.widgets import ImageAndHisto2DViewer
from deconv import DECONV_CONFIG_PATH, DECONV_MEASUREMENTS_DIR, DEFAULT_DECONV_CONFIG
from deconv.cache import update_cache
from .psf_parameters_ui_dictionary import PSF_PARAMETERS_UI


_PREVIEW_TITLES = {
    "real": ("g_raw = input image (raw measurement)", "g = preprocessed g_raw + noise"),
    "synthetic": ("f_true = input image (ground truth)", "g_synth = H * f_true (+ noise)"),
}


def _compute_preview(config, mode):
    # lazy import: deconv.gui is imported before deconv.core (see main.py).
    from deconv.core.pipeline_operations import DeconvOperations
    return DeconvOperations.compute_preprocessing_preview(config, mode)


def _preview_errors(config):
    add_noise = config.get('add-noise', {})
    if add_noise.get('add_noise', False) and add_noise.get('sigma', 'None') == 'None':
        return ['Noise standard deviation is None, please define a value.']
    return []


class DeconvInputFilesSection(BaseInputFilesSection):
    CONFIG_PATH = DECONV_CONFIG_PATH
    DEFAULT_CONFIG = DEFAULT_DECONV_CONFIG
    UPDATE_CACHE_FN = staticmethod(update_cache)
    MEASUREMENTS_DIR = DECONV_MEASUREMENTS_DIR
    REAL_MODE_TEXT = "Work with real measurement"
    SYNTHETIC_MODE_TEXT = "Simulate with synthetic truth"
    DEFAULT_MODE_REAL = False   # deconv defaults to synthetic

    IMAGE_SLOT = FileSlot(
        toml_key="png", noun="png", dialog_filter="Image Files (*.png)",
        title_real="Path of the 2D image (measurement)",
        title_synthetic="Path of the 2D ground truth",
        preview=Preview(ImageAndHisto2DViewer, _compute_preview,
                        titles=_PREVIEW_TITLES, size=(900, 900), error_check=_preview_errors),
    )
    JSON_SLOT = FileSlot(
        toml_key="json", noun="json", dialog_filter="Parameters Files (*.json)",
        title_real="Path of the PSF parameters",
        title_synthetic="Path of the simulation parameters",
        editor=Editor(ui=PSF_PARAMETERS_UI, title_noun="PSF Parameters"),   # generic editor
    )
