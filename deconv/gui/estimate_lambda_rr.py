"""
Extra-button callback for the 'lambda_rr' parameter in deconv MCMC UI dict.

Computes the Fourier spectrum of the PSF (padded to image size) and lets
the user choose what percentage of frequencies to keep.
lambda_rr is set to |H_fft|² at the chosen percentile.
"""

from PyQt5.QtWidgets import QMessageBox

import torch

from common.in_out import load_json, load_png
from common.gui.reusable.frequency_cutoff_dialog import FrequencyCutoffDialog
from deconv.core.operations import compute_psf_from_params, _psf_to_fft_kernel


def estimate_lambda_rr(widget, update_cache_fn, load_toml_fn, config_path):
    """Estimate lambda_rr from the Fourier spectrum of the PSF."""
    config = load_toml_fn(config_path)
    missing = []

    json_path = config.get('input-paths', {}).get('json', 'None')
    if json_path == 'None' or not json_path:
        missing.append("PSF parameters file (.json) not selected.")
        psf_params = None
    else:
        try:
            psf_params = load_json(json_path)
        except Exception as e:
            missing.append(f"Cannot load .json file: {type(e).__name__}: {e}")
            psf_params = None

    if psf_params is not None:
        for key in ['sigma', 'kernel_size']:
            if key not in psf_params or psf_params[key] in (None, 'None'):
                missing.append(f"Missing PSF parameter '{key}' in .json.")

    png_path = config.get('input-paths', {}).get('png', 'None')
    if png_path == 'None' or not png_path:
        missing.append("Image file (.png) not selected (needed to determine image size).")
        image = None
    else:
        try:
            image = load_png(png_path)
        except Exception as e:
            missing.append(f"Cannot load .png file: {type(e).__name__}: {e}")
            image = None

    if missing:
        QMessageBox.warning(
            widget, "Cannot estimate lambda_rr",
            "The following parameters are required:\n\n"
            + "\n".join(f"  • {m}" for m in missing)
        )
        return

    try:
        H = compute_psf_from_params(psf_params)
        H_padded = _psf_to_fft_kernel(H, image.shape)
        H_fft = torch.fft.fft2(H_padded)
        S = H_fft.abs().flatten().sort(descending=True).values
    except Exception as e:
        QMessageBox.warning(widget, "Cannot estimate lambda_rr", f"{type(e).__name__}: {e}")
        return

    dialog = FrequencyCutoffDialog(
        widget,
        title="Estimate lambda_rr from PSF spectrum",
        info_text=(
            f"PSF kernel: {tuple(H.shape)}    "
            f"Image: {tuple(image.shape)}"
        ),
        S=S,
    )

    def _on_accepted():
        update_cache_fn(['algo-params', 'lambda_rr'], dialog.selected_lambda_rr())
        widget.update_ui_from_toml(config_path)

    dialog.accepted.connect(_on_accepted)
    dialog.show()
