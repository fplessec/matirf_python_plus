"""
Deconvolution input-files section: two generic FileSelectors (PNG image + JSON PSF
parameters) wired to the deconv problem, plus its preprocessing preview and PSF editor.
"""

from PyQt5.QtWidgets import QMessageBox

from common.gui.reusable import FileSelector, SelectorButton
from common.gui.specializable.control_window import BaseInputFilesSection
from common import DataMode
from deconv import DECONV_CONFIG_PATH, DECONV_MEASUREMENTS_DIR, DEFAULT_DECONV_CONFIG
from common.in_out import load_or_create_toml
from deconv.cache import update_cache


# ── secondary-button callbacks (problem-specific sub-windows) ───────────

def _open_preprocess(selector):
    from .qwidget_preprocess_viewer import DeconvPreprocessViewer
    try:
        viewer = DeconvPreprocessViewer(parent=selector)
    except Exception as e:
        config = load_or_create_toml(DECONV_CONFIG_PATH, default_config=DEFAULT_DECONV_CONFIG)
        errors = []
        if config['add-noise']['add_noise'] and config['add-noise']['sigma'] == 'None':
            errors.append('Noise standard deviation value is None, please define a value for this parameter.')
        if errors:
            e = Exception('\n-' + '\n-'.join(errors))
        QMessageBox.warning(
            selector, "Cannot preview the preprocessing",
            f"An unexpected error occurred while computing the preview:\n\n{type(e).__name__}: {e}")
        return
    selector.set_sub_window(viewer)


def _open_psf_editor(selector):
    from .psf_parameters_editor import PsfParametersEditor
    selector.set_sub_window(PsfParametersEditor(parent=selector))


# ── selector factories ─────────────────────────────────────────────────

def _make_png_selector(section):
    return FileSelector(
        section,
        noun="png",
        dialog_filter="Image Files (*.png)",
        toml_key="png",
        measurements_dir=DECONV_MEASUREMENTS_DIR,
        update_cache_fn=update_cache,
        title_real="Path of the 2D image (measurement)",
        title_synthetic="Path of the 2D ground truth",
        extra_button=SelectorButton(
            text="See preprocessed file",
            on_click=_open_preprocess,
            visible_when=section.are_both_file_selected,
        ),
    )


def _make_json_selector(section):
    return FileSelector(
        section,
        noun="json",
        dialog_filter="Parameters Files (*.json)",
        toml_key="json",
        measurements_dir=DECONV_MEASUREMENTS_DIR,
        update_cache_fn=update_cache,
        title_real="Path of the PSF parameters",
        title_synthetic="Path of the simulation parameters",
        extra_button=SelectorButton(
            text=lambda selected: "Modify .json file" if selected else "Create .json file",
            on_click=_open_psf_editor,
        ),
    )


# ── input files section for the deconv problem ──────────────────────────

class DeconvInputFilesSection(BaseInputFilesSection):
    """
    Deconvolution input files section.

    Selects a .png file (2D image or ground truth) and a .json PSF parameters file,
    with a real / synthetic mode toggle.
    """

    def _get_cached_mode(self) -> bool:
        try:
            mode = load_or_create_toml(
                DECONV_CONFIG_PATH, default_config=DEFAULT_DECONV_CONFIG
            )['input-paths']['mode']
            return mode == DataMode.REAL.value
        except (KeyError, FileNotFoundError):
            return False  # default is synthetic for deconv

    def _real_mode_text(self):
        return "Work with real measurement"

    def _synthetic_mode_text(self):
        return "Simulate with synthetic truth"

    def _create_image_selector(self):
        return _make_png_selector(self)

    def _create_json_selector(self):
        return _make_json_selector(self)

    def _on_switch_mode(self, is_mode_real):
        new_mode = DataMode.REAL.value if is_mode_real else DataMode.SYNTHETIC.value
        update_cache(['input-paths', 'mode'], new_mode)

    def _load_config_for_update(self, toml_path):
        return load_or_create_toml(toml_path, default_config=DEFAULT_DECONV_CONFIG)

    def _get_file_paths_from_config(self, config):
        return config['input-paths']['png'], config['input-paths']['json']
