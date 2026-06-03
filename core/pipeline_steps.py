from typing import Optional, Tuple

import torch

from base.enums import DataMode
from .operations import (
    compute_matirf_operator_from_params,
    apply_matirf_operator,
    estimate_delta_anisotropy_from_params,
)
from .preprocess_measurement import preprocess_measurement_stack
from .reconstruction_metrics import compute_all_metrics, optimal_scale
from in_out import load_tif, load_json


## loads json measurement params and extracts oper_params and add_noise_params from config:
def load_measurement_inputs(config: dict) -> Tuple[dict, dict, dict]:
    measurement_params = load_json(config['input-paths']['json'])
    oper_params = config['oper-params']
    add_noise_params = config['add-noise']
    return measurement_params, oper_params, add_noise_params


## real mode: loads the experimental measurement, preprocesses it and builds H:
def build_g_and_H_real(config: dict) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    measurement_params, oper_params, add_noise_params = load_measurement_inputs(config)
    g_raw = load_tif(config['input-paths']['tif'])
    g, measurement_params = preprocess_measurement_stack(
        g_raw, measurement_params, add_noise_params
    )
    H = compute_matirf_operator_from_params(measurement_params, oper_params)
    return g, H, measurement_params


## synthetic mode: loads f_true, builds H_synth, simulates g = H_synth @ f_true and preprocesses it:
def build_g_and_H_synthetic(config: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    measurement_params, oper_params, add_noise_params = load_measurement_inputs(config)
    f_true = load_tif(config['input-paths']['tif'])
    nz_from_tif = f_true.shape[0]
    # use nz from the tif (not from the GUI) to avoid shape mismatch H @ f_true
    synth_oper_params = dict(oper_params)
    synth_oper_params['nz'] = nz_from_tif
    H_synth = compute_matirf_operator_from_params(measurement_params, synth_oper_params)
    g = apply_matirf_operator(H_synth, f_true)
    g, _ = preprocess_measurement_stack(g, measurement_params, add_noise_params)
    return g, H_synth, f_true


## computes delta anisotropy and all quality metrics (PSNR, SSIM, MSE):
def evaluate_synthetic_metrics(f: torch.Tensor, f_true: torch.Tensor,
                                config: dict) -> Tuple[dict, float]:
    measurement_params, oper_params, _ = load_measurement_inputs(config)
    delta = estimate_delta_anisotropy_from_params(measurement_params, oper_params)
    metrics = compute_all_metrics(f, f_true, delta=delta)
    return metrics, delta


## computes the scaled difference f_true - alpha*f where alpha minimizes the distance:
def compute_synthetic_difference(f: torch.Tensor, f_true: torch.Tensor,
                                  config: dict) -> Tuple[torch.Tensor, float]:
    measurement_params, oper_params, _ = load_measurement_inputs(config)
    delta = estimate_delta_anisotropy_from_params(measurement_params, oper_params)
    alpha = optimal_scale(
        f.detach().cpu().numpy(),
        f_true.detach().cpu().numpy(),
        delta=delta,
    )
    diff = f_true - alpha * f
    return diff, alpha


## returns (raw, preprocessed) for real mode or (f_true, g_synth) for synthetic mode:
def compute_preprocessing_preview(config: dict, mode: DataMode) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == DataMode.REAL:
        measurement_params, _, add_noise_params = load_measurement_inputs(config)
        raw = load_tif(config['input-paths']['tif'])
        preprocessed, _ = preprocess_measurement_stack(
            raw, measurement_params, add_noise_params
        )
        return raw, preprocessed
    # synthetic: f_true vs g_synth = H @ f_true after preprocessing
    g_synth, _, f_true = build_g_and_H_synthetic(config)
    return f_true, g_synth
