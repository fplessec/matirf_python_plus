
from base.abstract_pipeline import AbstractPipeline
from base.enums import DataMode, PipelineState
from .reconstruction_result import ReconstructionResult
from .pipeline_steps import (
    build_g_and_H_real,
    build_g_and_H_synthetic,
    evaluate_synthetic_metrics,
    compute_synthetic_difference,
    load_measurement_inputs,
)


class MaTirfPipeline(AbstractPipeline):

    def _create_result(self):
        return ReconstructionResult()

    ## loads tif + json, builds H the MA-TIRF operator, instantiates the algorithm:
    def setup(self):
        from algorithms import ALGORITHMS
        if self.mode == DataMode.REAL:
            g, H, measurement_params = build_g_and_H_real(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = None
        else:
            g, H, f_true = build_g_and_H_synthetic(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = f_true
            measurement_params, _, _ = load_measurement_inputs(self.config)
        self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"]()

    ## stores f, computes metrics with delta anisotropy and diff f_true - alpha*f:
    def _on_algo_finished(self, f):
        self.result.f = f
        if self.mode == DataMode.SYNTHETIC:
            metrics, delta = evaluate_synthetic_metrics(
                self.result.f, self.result.f_true, self.config
            )
            self.result.metrics = metrics
            self.result.delta = delta
            diff, _alpha = compute_synthetic_difference(
                self.result.f, self.result.f_true, self.config
            )
            self.result.diff = diff
        self._running = False
        self._set_state(PipelineState.COMPLETED)
        self._emit("finished", self.result)

    ## delegates save/load to result_io:
    def save_results(self, save_dir):
        from .result_io import save_reconstruction
        save_reconstruction(self.result, save_dir, self.config, self.mode)
        self._emit("message", f"Saved reconstruction in {save_dir}")
    def load_results(self, directory):
        from .result_io import load_reconstruction
        self.result = load_reconstruction(directory, self.mode)
        if self.mode == DataMode.SYNTHETIC and self.result.f is not None and self.result.f_true is not None:
            diff, _alpha = compute_synthetic_difference(
                self.result.f, self.result.f_true, self.config
            )
            self.result.diff = diff
            if self.result.delta is None:
                _, delta = evaluate_synthetic_metrics(
                    self.result.f, self.result.f_true, self.config
                )
                self.result.delta = delta
        self._set_state(PipelineState.COMPLETED)

    ## checks that tif, json, nz, z0, zN and noise sigma are provided:
    def validate_config(self) -> list[str]:
        errors = []
        cfg = self.config
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