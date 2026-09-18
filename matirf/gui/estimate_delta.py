"""
Extra-button callback for the 'delta' parameter in MA-TIRF algorithm UI dicts.

This module lives separately from control_window.py and the algorithm UI
dictionaries to avoid circular imports:
    control_window → ALGORITHMS → adam_ui_dict → estimate_delta  (no cycle)

The callback signature (widget, update_cache_fn, load_toml_fn, config_path)
is the standard 'extra_button' callback interface defined by SimpleParameterWidget.
"""

from PyQt5.QtWidgets import QMessageBox

from common.in_out import load_json
from problems.matirf.physics import estimate_anisotropy_ratio


def estimate_delta(widget, update_cache_fn, load_toml_fn, config_path):
    """Estimate the delta parameter from current config and write it back."""
    _REQUIRED_MEASUREMENT_KEYS = ['n_medium', 'numerical_aperture', 'wavelength_nm']
    _REQUIRED_OPER_KEYS = ['nz', 'z0', 'zN']
    config = load_toml_fn(config_path)
    missing = []
    json_path = config.get('input-paths', {}).get('json', 'None')
    if json_path == 'None' or not json_path:
        missing.append("Measurement parameters file (.json) not selected.")
        measurement_params = None
    else:
        try:
            measurement_params = load_json(json_path)
        except Exception as e:
            missing.append(f"Cannot load .json file: {type(e).__name__}: {e}")
            measurement_params = None
    if measurement_params is not None:
        for key in _REQUIRED_MEASUREMENT_KEYS:
            if key not in measurement_params or measurement_params[key] in (None, 'None'):
                missing.append(f"Missing measurement parameter '{key}' in .json.")
    oper_params = config.get('oper-params', {})
    for key in _REQUIRED_OPER_KEYS:
        if key not in oper_params or oper_params[key] in (None, 'None'):
            missing.append(f"Missing operator parameter '{key}' (set it in the operator section).")
    if missing:
        QMessageBox.warning(
            widget, "Cannot estimate delta",
            "The following parameters are required:\n\n" + "\n".join(f"  • {m}" for m in missing)
        )
        return
    try:
        delta = round(estimate_anisotropy_ratio(
            nz=oper_params['nz'], z0=oper_params['z0'], zN=oper_params['zN'],
            n_medium=measurement_params['n_medium'],
            numerical_aperture=measurement_params['numerical_aperture'],
            wavelength_nm=measurement_params['wavelength_nm']), 4)
    except Exception as e:
        QMessageBox.warning(widget, "Cannot estimate delta", f"{type(e).__name__}: {e}")
        return
    update_cache_fn(['algo-params', 'delta'], float(delta))
    widget.update_ui_from_toml(config_path)
