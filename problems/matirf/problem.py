"""
The MA-TIRF inverse problem, declared.

In v1 this description was spread over six files: the package __init__ (features, paths,
default config), the pipeline (seven class attributes), the pipeline operations (g/H,
validation, metrics, preview), the physics, an algorithm mixin, and the GUI. Here it is one
object, and nothing else in the project needs to know what MA-TIRF is.

Note what is NOT here: no algorithm, no GUI, no pipeline. The six solvers work on this
problem because it provides a ForwardOperator, and for no other reason.
"""

import torch

from core import InverseProblem
from problems.matirf import (  # paths defined before this module is imported
    MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG, MATIRF_RESULTS_DIR, MATIRF_MEASUREMENTS_DIR,
)
from core import noise, normalization
from core.normalization import normalize_and_add_noise
from fileio import load_tif, save_tif, load_json
from .operator import MatirfOperator
from .ui import MATIRF_UI
from . import physics


# ── loading and preprocessing the measurement ─────────────────────────────────

def split_background(g: torch.Tensor, measurement_params: dict):
    """
    Separate the background stacks from the signal stacks, and subtract the background.

    A stack acquired ABOVE the objective's maximum angle receives no light, so what it
    records is the camera's background. Averaging those stacks estimates the background,
    which is then subtracted from the real measurements.

    This is why loading must happen BEFORE the operator is built: dropping the background
    stacks also drops their angles, and H has exactly one row per remaining angle. The
    updated parameter dict is returned so the caller can build H from it.
    """
    angles = list(measurement_params["angles_deg"])
    n_angles, n_stacks = len(angles), g.shape[0]
    assert n_angles == n_stacks, (
        f"\nWhile preprocessing the MA-TIRF measurement:\n"
        f"The number of angles ({n_angles}) and the number of stacks ({n_stacks}) do not match.\n\n"
        f"Please check that the number of stacks in the .tif matches the number of angles "
        f"declared in the .json measurement parameters."
    )

    _, theta_max = physics.angle_bounds(measurement_params)
    is_background = [angle > theta_max for angle in angles]

    if not any(is_background):
        return g, measurement_params

    background = torch.stack([g[i] for i, bg in enumerate(is_background) if bg]).mean(dim=0)
    signal = torch.stack([g[i] for i, bg in enumerate(is_background) if not bg])
    refined = dict(measurement_params)
    refined["angles_deg"] = [a for a, bg in zip(angles, is_background) if not bg]

    dropped = ", ".join(str(i + 1) for i, bg in enumerate(is_background) if bg)
    print(f"stack(s) {dropped} are above the maximum angle ({theta_max:.2f} deg); "
          f"used to estimate the background and removed.")
    return signal - background.unsqueeze(0), refined


def preprocess(g: torch.Tensor, measurement_params: dict, config: dict):
    """
    Background removal, then the normalization and the optional simulated noise.

    MA-TIRF is scale-ambiguous (see the operator's features): the absolute level carries no
    information, so normalizing loses nothing — it only fixes the unit the noise level and
    the data fidelity are expressed in (core/normalization.py).
    """
    g, measurement_params = split_background(g, measurement_params)
    return normalize_and_add_noise(g, config), measurement_params


def _refined_config(config: dict, measurement_params: dict) -> dict:
    """
    A copy of the config carrying the measurement parameters as preprocessing left them.

    The operator is built from `[_measurement]` when present, so background removal is
    reflected in H without re-reading (and re-trusting) the .json from disk.
    """
    refined = dict(config)
    refined["_measurement"] = measurement_params
    return refined


def load_measurement(config: dict):
    """REAL mode: read the acquired stacks, preprocess them, and refine the config."""
    measurement_params = load_json(config["input-paths"]["json"])
    g, measurement_params = preprocess(load_tif(config["input-paths"]["tif"]),
                                       measurement_params, config)
    return g, _refined_config(config, measurement_params)


def load_truth(config: dict):
    """
    SYNTHETIC mode: read the known object f_true.

    The truth's own depth count wins over the configured `nz`: simulating requires an H
    whose columns match the slices actually present in the file, whatever the user typed.
    """
    f_true = load_tif(config["input-paths"]["tif"])
    measurement_params = load_json(config["input-paths"]["json"])
    refined = _refined_config(config, measurement_params)
    refined["oper-params"] = {**config.get("oper-params", {}), "nz": f_true.shape[0]}
    return f_true, refined


def simulate(operator, f_true: torch.Tensor, config: dict) -> torch.Tensor:
    """SYNTHETIC mode: g = H f_true, then the same normalization and noise a real one gets."""
    return normalize_and_add_noise(operator.apply(f_true), config)


def build_operator(config: dict) -> MatirfOperator:
    """
    Build the operator from the config.

    When a loader has already run, it left the post-preprocessing measurement parameters
    under `_measurement`, and those are used — so background removal is reflected in H
    without re-reading (and re-trusting) the .json from disk. Otherwise the operator reads
    the .json itself, which is what the GUI does to preview an operator before any run.
    """
    measurement = config.get("_measurement")
    if measurement is None:
        return MatirfOperator.from_config(config)
    return MatirfOperator.from_measurement(measurement, config.get("oper-params", {}))


# ── validation ────────────────────────────────────────────────────────────────

def validate(config: dict) -> list:
    """
    Every missing or unusable setting, as one message each.

    Read with `.get` and compared against the string "None" throughout: a section can be
    empty or a value stored as "None" right after `matirf reset`, and an unset parameter
    must be reported as a problem rather than crash on a missing key.
    """
    errors = []
    paths = config.get("input-paths", {})
    oper = config.get("oper-params", {})
    add_noise = config.get("add-noise", {})

    if paths.get("tif", "None") == "None":
        errors.append("Input file (TIF): not provided")
    if paths.get("json", "None") == "None":
        errors.append("Measurement parameters file (JSON): not provided")
    for key, label in (("nz", "number of z slices"), ("z0", "shallowest depth"),
                       ("zN", "deepest depth")):
        if oper.get(key, "None") == "None":
            errors.append(f"Operator parameter '{key}' ({label}): not set")

    ## `normalize` is required like the rest, but it is a CHECKBOX, and a checkbox the user
    ## never touched is not written to the TOML at all. So "unset" here means absent or
    ## null — and explicitly NOT `False`, which is a perfectly valid answer. A plain
    ## truthiness test would report every "no, do not normalize" as a missing setting.
    if oper.get("normalize", "None") in (None, "None", "null"):
        errors.append("Operator parameter 'normalize' (normalize the operator): not set")
    errors.extend(noise.validate(add_noise))
    errors.extend(normalization.validate(config.get("input-paths", {}).get("normalization")))
    return errors


# ── the declaration ───────────────────────────────────────────────────────────

MATIRF = InverseProblem(
    name="matirf",
    operator_class=MatirfOperator,
    build_operator=build_operator,
    load_measurement=load_measurement,
    load_truth=load_truth,
    simulate=simulate,
    validate=validate,
    save_image=save_tif,
    load_image=load_tif,
    image_extension="TIF",
    raw_path_key="tif",
    config_path=MATIRF_CONFIG_PATH,
    default_config=DEFAULT_MATIRF_CONFIG,
    results_dir=MATIRF_RESULTS_DIR,
    measurements_dir=MATIRF_MEASUREMENTS_DIR,
    ui=MATIRF_UI,
    description="Multi-angle TIRF: recover a 3D object from stacks measured at several "
                "incidence angles, each probing a different depth.",
)
