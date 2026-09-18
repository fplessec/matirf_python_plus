"""
Base class for problem-specific pipeline operations.

Each inverse problem (matirf, deconv, ...) subclasses PipelineOperations
and implements these static methods:
    - build_g_H_real(config)                          → (g, H)
    - build_g_H_synthetic(config)                     → (g, H, f_true)
    - validate_config(config)                         → list of error strings
    - compute_synthetic_outputs(result, config, feat) → fills result in-place
    - compute_preprocessing_preview(config, mode)     → (left, right) tensors
"""


class PipelineOperations:

    @staticmethod
    def build_g_H_real(config):
        raise NotImplementedError

    @staticmethod
    def build_g_H_synthetic(config):
        raise NotImplementedError

    @staticmethod
    def validate_config(config):
        raise NotImplementedError

    @staticmethod
    def compute_synthetic_outputs(result, config, features):
        raise NotImplementedError

    @staticmethod
    def compute_preprocessing_preview(config, mode):
        raise NotImplementedError
