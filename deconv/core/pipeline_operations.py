"""Deconvolution pipeline operations — builds g/H, validates config, computes metrics."""

from typing import Tuple

from common.core.enums import DataMode
from common.core.pipeline_operations import PipelineOperations
from common.core.metrics import compute_all_metrics
from common.add_noise import add_noise_to_measurement
from common.in_out import load_png, load_json
from .operations import compute_psf_from_params, apply_psf


def _load_measurement_inputs(config: dict) -> Tuple[dict, dict]:
    psf_params = load_json(config['input-paths']['json'])
    add_noise_params = config['add-noise']
    return psf_params, add_noise_params


class DeconvOperations(PipelineOperations):

    # ── build g and H ─────────────────────────────────────────────────────

    @staticmethod
    def build_g_H_real(config):
        psf_params, add_noise_params = _load_measurement_inputs(config)
        g = load_png(config['input-paths']['png'])
        if add_noise_params.get('add_noise', False):
            g = add_noise_to_measurement(g, add_noise_params)
        H = compute_psf_from_params(psf_params)
        return g, H

    @staticmethod
    def build_g_H_synthetic(config):
        psf_params, add_noise_params = _load_measurement_inputs(config)
        f_true = load_png(config['input-paths']['png'])
        H = compute_psf_from_params(psf_params)
        g = apply_psf(H, f_true)
        if add_noise_params.get('add_noise', False):
            g = add_noise_to_measurement(g, add_noise_params)
        return g, H, f_true

    # ── validation ────────────────────────────────────────────────────────

    @staticmethod
    def validate_config(config) -> list[str]:
        errors = []
        cfg = config
        if cfg['algorithm'] == 'None':
            errors.append("Algorithm: not selected")
        if cfg['input-paths']['png'] == 'None':
            errors.append("Input file (PNG): not provided")
        if cfg['input-paths']['json'] == 'None':
            errors.append("PSF parameters file (JSON): not provided")
        if cfg['add-noise'].get('add_noise', False) and cfg['add-noise'].get('sigma', 'None') == 'None':
            errors.append("Noise sigma: required when 'add noise' is enabled")
        return errors

    # ── synthetic outputs ─────────────────────────────────────────────────

    @staticmethod
    def compute_synthetic_outputs(result, config, features):
        result.metrics = compute_all_metrics(result.f, result.f_true, features=features)
        result.diff = result.f_true - result.f

    # ── preview ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_preprocessing_preview(config, mode):
        if mode == DataMode.REAL:
            _, add_noise_params = _load_measurement_inputs(config)
            raw = load_png(config['input-paths']['png'])
            if add_noise_params.get('add_noise', False):
                preprocessed = add_noise_to_measurement(raw, add_noise_params)
            else:
                preprocessed = raw
            return raw, preprocessed
        g_synth, _, f_true = DeconvOperations.build_g_H_synthetic(config)
        return f_true, g_synth
