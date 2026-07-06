from .qwidget_tif_file_selector import TifFileSelector
from .qwidget_json_file_selector import JsonFileSelector
from common.gui.specializable.control_window import BaseInputFilesSection
from common import DataMode
from matirf import MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG
from common.in_out import load_or_create_toml
from matirf.cache import update_cache


class InputFilesSection(BaseInputFilesSection):
    """
    MA-TIRF input files section.

    Selects a .tif file (MA-TIRF stack or 3D synthetic truth) and a .json
    measurement parameters file, with a real / synthetic mode toggle.
    """

    def _get_cached_mode(self) -> bool:
        try:
            mode = load_or_create_toml(
                MATIRF_CONFIG_PATH, default_config=DEFAULT_MATIRF_CONFIG
            )['input-paths']['mode']
            return mode == DataMode.REAL.value
        except (KeyError, FileNotFoundError):
            return True  # default is real

    def _real_mode_text(self):
        return "Work with real MA-TIRF measurement"

    def _synthetic_mode_text(self):
        return "Simulate measurement with synthetic truth"

    def _create_image_selector(self):
        return TifFileSelector(parent=self)

    def _create_json_selector(self):
        return JsonFileSelector(parent=self)

    def _on_switch_mode(self, is_mode_real):
        new_mode = DataMode.REAL.value if is_mode_real else DataMode.SYNTHETIC.value
        update_cache(['input-paths', 'mode'], new_mode)

    def _load_config_for_update(self, toml_path):
        return load_or_create_toml(toml_path, default_config=DEFAULT_MATIRF_CONFIG)

    def _get_file_paths_from_config(self, config):
        return config['input-paths']['tif'], config['input-paths']['json']
