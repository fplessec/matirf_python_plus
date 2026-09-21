"""
The '[noise-model]' section: which likelihood the reconstruction assumes, and at what level.

It is a property of the MEASUREMENT, not of any algorithm — the detector's noise, as H is
the optics' blur — which is why it has its own section instead of being an algorithm
parameter. A solver only declares the models it can minimize; choosing one it cannot is
refused when the run starts (see pipeline.validate_noise_model).

    fidelity     the noise model, i.e. the data fidelity D (solvers/fidelities)
    parameters   where its level (a, b) comes from:
                    estimated   measured on the preprocessed g — live, below
                    oracle      the values '[add-noise]' used — synthetic mode only
                    manual      a and b as typed below
                    unscaled    a = b = 1: v1's D = 1/2 mean (Hf - g)^2 in the Gaussian case
    a, b         shown only when manual, and only those the chosen model uses

The formula line, first in the section, shows D with the values it will use. With
'estimated' it computes them live — from the same preprocessed g "See preprocessed file"
shows — and updates whenever anything that changes g does (files, normalization, operator
parameters, added noise). A noise found negligible is announced there, with the switch to
the unscaled D it causes (see pipeline.noise_parameters).
"""

import json

from gui.base.base_section_qgroup import BaseSectionQGroup

_GAUSSIAN, _POISSON, _MIXED = ("L2 (Gaussian noise)", "KL divergence (Poisson noise)",
                               "Poisson-Gaussian")

NOISE_MODEL_UI = {
    "fidelity": {
        "title": "Data fidelity (noise model)",
        "type": "option",
        "param_info": {"options_list": [_GAUSSIAN, _POISSON, _MIXED]},
    },
    "parameters": {
        "title": "Noise level (a, b)",
        "type": "option",
        "param_info": {"options_list": ["estimated", "oracle", "manual", "unscaled"]},
    },
    "a": {
        "title": "Photon noise a = 1 / N",
        "type": "value",
        "depends_on": {"fidelity": [_POISSON, _MIXED], "parameters": "manual"},
        "param_info": {"dtype": float, "unit": "", "latex_name": "a", "default": 0.01},
    },
    "b": {
        "title": "Read noise b = σ²",
        "type": "value",
        "depends_on": {"fidelity": [_GAUSSIAN, _MIXED], "parameters": "manual"},
        "param_info": {"dtype": float, "unit": "", "latex_name": "b", "default": 1e-4},
    },
}


def noise_model_formula(values: dict, config: dict, problem=None) -> str:
    """
    D with the (a, b) the run will use.

    `config` is the whole cached config; this section's current `values` replace its
    '[noise-model]' part. With 'estimated', g is computed through `problem.preview` — the
    code path a run takes — and the estimate is cached on everything g depends on.
    """
    from pipeline import noise_parameters
    config = {**config, "noise-model": dict(values)}
    source = values.get("parameters", "estimated")
    try:
        g = _preprocessed(problem, config) if source == "estimated" else None
        cls, a, b, note = noise_parameters(config, g)
    except _Unavailable as reason:
        from pipeline import fidelity_class
        return fidelity_class(config).formula + rf",\quad \text{{({reason})}}"
    except (TypeError, ValueError) as error:
        from pipeline import fidelity_class
        return fidelity_class(config).formula + rf",\quad \text{{({_plain(error)})}}"
    shown = cls.from_noise(a, b).latex()
    if "negligible" in note:
        return shown + rf"\quad \text{{({source} noise negligible: unscaled, as in v1)}}"
    return shown + rf"\quad \text{{({source})}}"


class _Unavailable(Exception):
    """g cannot be computed yet: files or parameters missing."""


_G_CACHE = {}


def _preprocessed(problem, config):
    if problem is None:
        raise _Unavailable("estimated when the run starts")
    relevant = {key: config.get(key) for key in ("input-paths", "oper-params", "add-noise")}
    key = (problem.name, json.dumps(relevant, sort_keys=True, default=str))
    if key not in _G_CACHE:
        if problem.validate(config):
            raise _Unavailable("set the input files and parameters to estimate a and b")
        try:
            _G_CACHE.clear()                 # one measurement at a time is enough
            _G_CACHE[key] = problem.preview(config)[1]
        except Exception as error:
            raise _Unavailable(f"g unavailable: {_plain(error)}")
    return _G_CACHE[key]


def _plain(error) -> str:
    """An exception as text mathtext can show (no $, _, ^ or braces)."""
    text = str(error).splitlines()[0][:60] if str(error) else type(error).__name__
    return "".join(c for c in text if c not in "$_^{}\\%#&~")


class NoiseModelSection(BaseSectionQGroup):
    """Shared section choosing the noise model — the data fidelity — of the reconstruction."""

    title = "Noise model (data fidelity)"
    params_ui_dict = NOISE_MODEL_UI
    toml_section_key = 'noise-model'
    formula = staticmethod(noise_model_formula)
