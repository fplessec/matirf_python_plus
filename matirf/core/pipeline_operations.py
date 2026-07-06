"""MA-TIRF pipeline operations — builds g/H, validates config, computes metrics."""

from typing import Tuple

import torch

from common.core.enums import DataMode
from common.core.pipeline_operations import PipelineOperations
from common.core.metrics import compute_all_metrics, optimal_scale
from common.in_out import load_tif, load_json
from .operations import (
    compute_matirf_operator_from_params,
    apply_matirf_operator,
    estimate_delta_anisotropy_from_params,
)
from .preprocess_measurement import preprocess_measurement_stack


def _load_measurement_inputs(config: dict) -> Tuple[dict, dict, dict]:
    measurement_params = load_json(config['input-paths']['json'])
    oper_params = config['oper-params']
    add_noise_params = config['add-noise']
    return measurement_params, oper_params, add_noise_params


class MaTirfOperations(PipelineOperations):

    # ── build g and H ─────────────────────────────────────────────────────

    @staticmethod
    def build_g_H_real(config):
        measurement_params, oper_params, add_noise_params = _load_measurement_inputs(config)
        g_raw = load_tif(config['input-paths']['tif'])
        g, measurement_params = preprocess_measurement_stack(
            g_raw, measurement_params, add_noise_params
        )
        H = compute_matirf_operator_from_params(measurement_params, oper_params)
        return g, H

    @staticmethod
    def build_g_H_synthetic(config):
        measurement_params, oper_params, add_noise_params = _load_measurement_inputs(config)
        f_true = load_tif(config['input-paths']['tif'])
        nz_from_tif = f_true.shape[0]
        synth_oper_params = dict(oper_params)
        synth_oper_params['nz'] = nz_from_tif
        H_synth = compute_matirf_operator_from_params(measurement_params, synth_oper_params)
        g = apply_matirf_operator(H_synth, f_true)
        g, _ = preprocess_measurement_stack(g, measurement_params, add_noise_params)
        return g, H_synth, f_true

    # ── validation ────────────────────────────────────────────────────────

    @staticmethod
    def validate_config(config) -> list[str]:
        errors = []
        cfg = config
        if cfg['algorithm'] == 'None':
            errors.append("Algorithm: not selected")
        if cfg['input-paths']['tif'] == 'None':
            errors.append("Input file (TIF): not provided")
        if cfg['input-paths']['json'] == 'None':
            errors.append("Measurement parameters file (JSON): not provided")
        if cfg['oper-params']['nz'] == 'None':
            errors.append("Operator parameter 'nz': not set")
        if cfg['oper-params']['z0'] == 'None':
            errors.append("Operator parameter 'z0': not set")
        if cfg['oper-params']['zN'] == 'None':
            errors.append("Operator parameter 'zN': not set")
        if cfg['add-noise']['add_noise'] and cfg['add-noise']['sigma'] == 'None':
            errors.append("Noise sigma: required when 'add noise' is enabled")
        return errors

    # ── synthetic outputs ─────────────────────────────────────────────────

    @staticmethod
    def compute_synthetic_outputs(result, config, features):
        measurement_params, oper_params, _ = _load_measurement_inputs(config)
        delta = estimate_delta_anisotropy_from_params(measurement_params, oper_params)
        result.metrics = compute_all_metrics(result.f, result.f_true, features=features, delta=delta)
        result.delta = delta
        alpha = optimal_scale(result.f, result.f_true)
        result.diff = result.f_true - alpha * result.f

    # ── preview ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_preprocessing_preview(config, mode):
        if mode == DataMode.REAL:
            measurement_params, _, add_noise_params = _load_measurement_inputs(config)
            raw = load_tif(config['input-paths']['tif'])
            preprocessed, _ = preprocess_measurement_stack(
                raw, measurement_params, add_noise_params
            )
            return raw, preprocessed
        g_synth, _, f_true = MaTirfOperations.build_g_H_synthetic(config)
        return f_true, g_synth
