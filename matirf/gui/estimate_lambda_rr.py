"""
Extra-button callback for the 'lambda_rr' parameter in MA-TIRF MCMC UI dict.

Computes the SVD of the MA-TIRF operator H, displays the singular value
spectrum, and lets the user pick which singular value to use as cutoff.
lambda_rr is set to s_k² (the square of the chosen singular value).
"""

from PyQt5.QtWidgets import QMessageBox

import torch

from common.in_out import load_json
from common.gui.reusable.singular_value_picker_dialog import SingularValuePickerDialog
from matirf.core.operations import compute_matirf_operator_from_params


def estimate_lambda_rr(widget, update_cache_fn, load_toml_fn, config_path):
    """Estimate lambda_rr from the singular value spectrum of H."""
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

    _REQUIRED_MEASUREMENT_KEYS = [
        'angles_deg', 'n_glass', 'n_medium', 'n_oil',
        'numerical_aperture', 'wavelength_nm', 'beam_divergence_deg',
    ]
    if measurement_params is not None:
        for key in _REQUIRED_MEASUREMENT_KEYS:
            if key not in measurement_params or measurement_params[key] in (None, 'None'):
                missing.append(f"Missing measurement parameter '{key}' in .json.")

    oper_params = config.get('oper-params', {})
    for key in ['nz', 'z0', 'zN', 'normalize']:
        if key not in oper_params or oper_params[key] in (None, 'None'):
            missing.append(f"Missing operator parameter '{key}'.")

    if missing:
        QMessageBox.warning(
            widget, "Cannot estimate lambda_rr",
            "The following parameters are required:\n\n"
            + "\n".join(f"  • {m}" for m in missing)
        )
        return

    try:
        H = compute_matirf_operator_from_params(measurement_params, oper_params)
        S = torch.linalg.svdvals(H)
    except Exception as e:
        QMessageBox.warning(widget, "Cannot estimate lambda_rr", f"{type(e).__name__}: {e}")
        return

    dialog = SingularValuePickerDialog(
        widget,
        title="Estimate lambda_rr from SVD of H",
        info_text=f"H shape: {tuple(H.shape)}    cond(H): {S[0]/S[-1]:.1f}",
        spectrum_label_prefix="s",
        S=S,
    )

    def _on_accepted():
        update_cache_fn(['algo-params', 'lambda_rr'], dialog.selected_lambda_rr())
        widget.update_ui_from_toml(config_path)

    dialog.accepted.connect(_on_accepted)
    dialog.show()
