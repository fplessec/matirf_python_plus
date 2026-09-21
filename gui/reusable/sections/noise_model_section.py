"""
The '[noise-model]' section: which likelihood the reconstruction assumes, and at what level.

It is a property of the MEASUREMENT, not of any algorithm — the detector's noise, as H is
the optics' blur — which is why it has its own section instead of being an algorithm
parameter. A solver only declares the models it can minimize; choosing one it cannot is
refused when the run starts (see pipeline.validate_noise_model).

    fidelity     the noise model, i.e. the data fidelity D (solvers/fidelities)
    parameters   where its level (a, b) comes from:
                    estimated   measured on the preprocessed g when the run starts
                    oracle      the values '[add-noise]' used — synthetic mode only
                    manual      a and b as typed below
    a, b         shown only when manual, and only those the chosen model uses

The formula line shows D with the values it will use.
"""

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
        "param_info": {"options_list": ["estimated", "oracle", "manual"]},
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


def noise_model_formula(values: dict, config: dict) -> str:
    """D, with the (a, b) it will use — or a note saying they are estimated at run time."""
    from core import DataMode, noise
    from solvers.fidelities import DATA_FIDELITY_REGISTRY, GaussianFidelity

    cls = DATA_FIDELITY_REGISTRY.get(values.get("fidelity"), GaussianFidelity)
    source = values.get("parameters", "estimated")
    if source == "manual":
        try:
            return cls.from_noise(float(values.get("a") or 0), float(values.get("b") or 0)).latex()
        except (TypeError, ValueError):
            return cls.formula + r",\quad \text{(enter a and b)}"
    if source == "oracle":
        if config.get("input-paths", {}).get("mode") != DataMode.SYNTHETIC.value:
            return cls.formula + r",\quad \text{(oracle: synthetic mode only)}"
        model = noise.noise_model(config.get("add-noise", {}))
        return cls.from_noise(model.a, model.b).latex()
    used = r",\ ".join(cls.noise_parameters)
    return cls.formula + rf",\quad {used}\ \text{{estimated from }} g"


class NoiseModelSection(BaseSectionQGroup):
    """Shared section choosing the noise model — the data fidelity — of the reconstruction."""

    title = "Noise model (data fidelity)"
    params_ui_dict = NOISE_MODEL_UI
    toml_section_key = 'noise-model'
    formula = staticmethod(noise_model_formula)
