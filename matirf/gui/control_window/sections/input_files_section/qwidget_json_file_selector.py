"""
MA-TIRF .json (measurement parameters) selector: a generic FileSelector wired to the
MA-TIRF problem, plus its required-keys validator and its parameters editor.
"""

from common.gui.reusable import FileSelector, SelectorButton
from common.in_out import load_json
from matirf import MATIRF_MEASUREMENTS_DIR
from matirf.cache import update_cache
from .measurement_parameters_editor import MeasurementParametersEditor, MEASUREMENT_PARAMETERS_UI


def check_measurement_parameters_file_format(filepath):
    """Ensure the selected JSON has all required measurement-parameter keys."""
    dictionary = load_json(filepath)
    missing_keys = [key for key in MEASUREMENT_PARAMETERS_UI if key not in dictionary]
    if missing_keys:
        print(f"Failed to select the following file: '{filepath}'\n"
              f"-> Invalid measurement parameters file format. This file is missing required key(s): "
              f"{', '.join(missing_keys)}")
    return not missing_keys


def _open_editor(selector):
    selector.set_sub_window(MeasurementParametersEditor(parent=selector))


def make_json_selector(section):
    """Build the MA-TIRF measurement-parameters FileSelector for the given input-files section."""
    return FileSelector(
        section,
        noun="json",
        dialog_filter="Parameters Files (*.json)",
        toml_key="json",
        measurements_dir=MATIRF_MEASUREMENTS_DIR,
        update_cache_fn=update_cache,
        title_real="Path of the measurement parameters",
        title_synthetic="Path of the simulated parameters",
        validate_fn=check_measurement_parameters_file_format,
        extra_button=SelectorButton(
            text=lambda selected: "Modify .json file" if selected else "Create .json file",
            on_click=_open_editor,
        ),
    )
