"""
MA-TIRF .tif selector: a generic FileSelector wired to the MA-TIRF problem, plus its
problem-specific "See preprocessed file" preview (with MA-TIRF-specific error reporting).
"""

from PyQt5.QtWidgets import QMessageBox

from common.gui.reusable import FileSelector, SelectorButton
from common.in_out import load_or_create_toml
from matirf import MATIRF_MEASUREMENTS_DIR, MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG
from matirf.cache import update_cache
from .qwidget_tif_file_preprocess_viewer import TifFilePreprocessViewer


def _open_preprocess(selector):
    """Open the MA-TIRF preprocessing preview, reporting problem-specific config errors."""
    try:
        viewer = TifFilePreprocessViewer(parent=selector)
    except AssertionError as e:
        QMessageBox.warning(
            selector, "Cannot preview the preprocessing",
            "The preview could not be computed because the measurement files "
            f"are inconsistent:\n{e}\n\n")
        return
    except Exception as e:
        config = load_or_create_toml(MATIRF_CONFIG_PATH, default_config=DEFAULT_MATIRF_CONFIG)
        # .get everywhere: these sections may be empty right after a cache reset, and this
        # error handler must never itself raise while reporting the real error.
        add_noise = config.get('add-noise', {})
        oper = config.get('oper-params', {})
        errors = []
        if add_noise.get('add_noise', False) and add_noise.get('sigma', 'None') == 'None':
            errors.append('Noise standard deviation value is None, please define a value for this parameter.')
        if oper.get('nz', 'None') == 'None':
            errors.append('Number of cuts on z is None, please define a value for this parameter')
        if oper.get('z0', 'None') == 'None':
            errors.append('Smallest depth is None, please define a value for this parameter')
        if oper.get('zN', 'None') == 'None':
            errors.append('Largest depth is None, please define a value for this parameter')
        if errors:
            e = Exception('\n-' + '\n-'.join(errors))
        QMessageBox.warning(
            selector, "Cannot preview the preprocessing",
            f"An unexpected error occurred while computing the preview:\n\n{type(e).__name__}: {e}")
        return
    selector.set_sub_window(viewer)


def make_tif_selector(section):
    """Build the MA-TIRF .tif FileSelector for the given input-files section."""
    return FileSelector(
        section,
        noun="tif",
        dialog_filter="Image Files (*.tif *.tiff)",
        toml_key="tif",
        measurements_dir=MATIRF_MEASUREMENTS_DIR,
        update_cache_fn=update_cache,
        title_real="Path of the MA-TIRF image stack",
        title_synthetic="Path of the 3D object (synthetic truth)",
        extra_button=SelectorButton(
            text="See preprocessed file",
            on_click=_open_preprocess,
            visible_when=section.are_both_file_selected,   # shown only when both files are selected
        ),
    )
