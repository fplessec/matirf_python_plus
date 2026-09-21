"""
The 2D deconvolution inverse problem, declared.

Compare this file with problems/matirf/problem.py: same shape, far less content, because
deconvolution has no background stacks to remove and no grid to reconcile. Loading is just
reading a PNG. That asymmetry is the point — the framework does not force a simple problem
to carry machinery it does not need.
"""

import torch

from core import InverseProblem
from problems.deconv import (  # paths defined before this module is imported
    DECONV_CONFIG_PATH, DEFAULT_DECONV_CONFIG, DECONV_RESULTS_DIR, DECONV_MEASUREMENTS_DIR,
)
from core import noise
from fileio import load_png, save_png
from .operator import DeconvOperator
from .ui import DECONV_UI


def _add_configured_noise(g: torch.Tensor, config: dict) -> torch.Tensor:
    """Apply the noise the user asked for, if any. The section may be empty right after a
    cache reset, which means no noise rather than a crash (see core/noise.py)."""
    return noise.add_noise_to_measurement(g, config.get("add-noise", {}))


def load_measurement(config: dict) -> torch.Tensor:
    """REAL mode: the blurred image, as acquired, plus any simulated extra noise."""
    return _add_configured_noise(load_png(config["input-paths"]["png"]), config)


def load_truth(config: dict) -> torch.Tensor:
    """SYNTHETIC mode: the sharp image whose blurred version we will try to invert."""
    return load_png(config["input-paths"]["png"])


def simulate(operator, f_true: torch.Tensor, config: dict) -> torch.Tensor:
    """SYNTHETIC mode: blur the truth, then add the configured noise."""
    return _add_configured_noise(operator.apply(f_true), config)


def validate(config: dict) -> list:
    """
    Every missing or unusable setting, as one message each.

    Read with `.get` and compared against "None" throughout, so a config emptied by
    `deconv reset` produces a readable list instead of a KeyError.
    """
    errors = []
    paths = config.get("input-paths", {})
    add_noise = config.get("add-noise", {})

    if paths.get("png", "None") == "None":
        errors.append("Input file (PNG): not provided")
    if paths.get("json", "None") == "None":
        errors.append("PSF parameters file (JSON): not provided")
    errors.extend(noise.validate(add_noise))
    return errors


DECONV = InverseProblem(
    name="deconv",
    operator_class=DeconvOperator,
    load_measurement=load_measurement,
    load_truth=load_truth,
    simulate=simulate,
    validate=validate,
    save_image=save_png,
    load_image=load_png,
    image_extension="png",
    raw_path_key="png",
    config_path=DECONV_CONFIG_PATH,
    default_config=DEFAULT_DECONV_CONFIG,
    results_dir=DECONV_RESULTS_DIR,
    measurements_dir=DECONV_MEASUREMENTS_DIR,
    ui=DECONV_UI,
    description="2D deconvolution: recover a sharp image from one blurred by a known "
                "point-spread function.",
)
