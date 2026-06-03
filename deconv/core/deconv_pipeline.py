from base.abstract_pipeline import AbstractPipeline
from base.enums import DataMode, PipelineState
from .deconv_result import DeconvResult
from .pipeline_steps import (
    build_g_and_H_real,
    build_g_and_H_synthetic,
    evaluate_synthetic_metrics,
    compute_synthetic_difference,
)


class DeconvPipeline(AbstractPipeline):

    def _create_result(self):
        return DeconvResult()

    ## loads png + json, builds gaussian PSF, instantiates the algorithm:
    def setup(self):
        from deconv.algorithms import ALGORITHMS
        if self.mode == DataMode.REAL:
            g, H = build_g_and_H_real(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = None
        else:
            g, H, f_true = build_g_and_H_synthetic(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = f_true
        self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"]()

    ## stores f, computes PSNR/SSIM/MSE metrics and diff f_true - f:
    def _on_algo_finished(self, f):
        self.result.f = f
        if self.mode == DataMode.SYNTHETIC:
            self.result.metrics = evaluate_synthetic_metrics(
                self.result.f, self.result.f_true
            )
            self.result.diff = compute_synthetic_difference(
                self.result.f, self.result.f_true
            )
        self._running = False
        self._set_state(PipelineState.COMPLETED)
        self._emit("finished", self.result)

    ## delegates save/load to result_io:
    def save_results(self, save_dir):
        from .result_io import save_deconv
        save_deconv(self.result, save_dir, self.config, self.mode)
        self._emit("message", f"Saved deconvolution in {save_dir}")
    def load_results(self, directory):
        from .result_io import load_deconv
        self.result = load_deconv(directory, self.mode)
        if self.mode == DataMode.SYNTHETIC and self.result.f is not None and self.result.f_true is not None:
            self.result.diff = compute_synthetic_difference(
                self.result.f, self.result.f_true
            )
        self._set_state(PipelineState.COMPLETED)

    ## checks that png, json and noise sigma are provided:
    def validate_config(self) -> list[str]:
        errors = []
        cfg = self.config
        if cfg['algorithm'] == 'None':
            errors.append("Algorithm: not selected")
        if cfg['input-paths']['png'] == 'None':
            errors.append("Input file (PNG): not provided")
        if cfg['input-paths']['json'] == 'None':
            errors.append("PSF parameters file (JSON): not provided")
        if cfg['add-noise'].get('add_noise', False) and cfg['add-noise'].get('sigma', 'None') == 'None':
            errors.append("Noise sigma: required when 'add noise' is enabled")
        return errors
