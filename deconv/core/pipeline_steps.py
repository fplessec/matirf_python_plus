from typing import Tuple

import torch

from base import DataMode
from base.add_noise import add_noise_to_measurement
from .operations import compute_psf_from_params, apply_psf
from .metrics import compute_all_metrics
from deconv.in_out import load_png, load_json


## loads PSF params from json and extracts add_noise_params from config:
def load_measurement_inputs(config: dict) -> Tuple[dict, dict]:
    psf_params = load_json(config['input-paths']['json'])
    add_noise_params = config['add-noise']
    return psf_params, add_noise_params


## real mode: loads the blurred png, builds the PSF and optionally adds noise:
def build_g_and_H_real(config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    psf_params, add_noise_params = load_measurement_inputs(config)
    g = load_png(config['input-paths']['png'])
    if add_noise_params.get('add_noise', False):
        g = add_noise_to_measurement(g, add_noise_params)
    H = compute_psf_from_params(psf_params)
    return g, H


## synthetic mode: loads f_true, builds H, simulates g = H * f_true and optionally adds noise:
def build_g_and_H_synthetic(config: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    psf_params, add_noise_params = load_measurement_inputs(config)
    f_true = load_png(config['input-paths']['png'])
    H = compute_psf_from_params(psf_params)
    g = apply_psf(H, f_true)
    if add_noise_params.get('add_noise', False):
        g = add_noise_to_measurement(g, add_noise_params)
    return g, H, f_true


## computes all quality metrics (PSNR, SSIM, MSE) between f and f_true:
def evaluate_synthetic_metrics(f: torch.Tensor, f_true: torch.Tensor) -> dict:
    return compute_all_metrics(f, f_true)


## computes the pixel-wise difference f_true - f:
def compute_synthetic_difference(f: torch.Tensor, f_true: torch.Tensor) -> torch.Tensor:
    return f_true - f


## returns (raw, noisy) for real mode or (f_true, g_synth) for synthetic mode:
def compute_preprocessing_preview(config: dict, mode: DataMode) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == DataMode.REAL:
        _, add_noise_params = load_measurement_inputs(config)
        raw = load_png(config['input-paths']['png'])
        if add_noise_params.get('add_noise', False):
            preprocessed = add_noise_to_measurement(raw, add_noise_params)
        else:
            preprocessed = raw
        return raw, preprocessed
    # mode == DataMode.SYNTHETIC
    g_synth, _, f_true = build_g_and_H_synthetic(config)
    return f_true, g_synth
