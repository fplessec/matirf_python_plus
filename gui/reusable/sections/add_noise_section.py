from core import noise
from gui.base.base_section_qgroup import BaseSectionQGroup


## The Poisson-Gaussian noise model of core/noise.py, as two independent switches:
##     photon noise only -> 100% Poisson,   read noise only -> 100% Gaussian,
##     both -> Poisson-Gaussian,            neither -> no noise.
## There is no separate "add noise" box: noise is on as soon as one of the two is ticked.
## Both parameters are in NORMALIZED units (see core/normalization.py): N is the photon count
## for an intensity of 1, sigma a fraction of it. The formula line under the section shows
## the resulting model and its (a, b) = (1/N, sigma^2) — the numbers the noise model uses.
ADD_NOISE_PARAMETERS_UI = {
        "poisson_noise": {
            "title": "Photon noise (Poisson)",
            "type": "bool",
            "param_info": {
                'default': False
            }
        },
        "photons": {
            "title": "Photons for an intensity of 1",
            "type": "value",
            "depends_on": {"poisson_noise": True},
            "param_info": {
                'dtype': float,
                'unit': 'photons',
                'latex_name': 'N',
                'default': 100.0
            }
        },
        "gaussian_noise": {
            "title": "Read noise (Gaussian)",
            "type": "bool",
            "param_info": {
                'default': False
            }
        },
        "sigma": {
            "title": "Read noise standard deviation",
            "type": "value",
            "depends_on": {"gaussian_noise": True},
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\sigma',
                'default': 0.01
            }
        },
        "seed": {
            "title": "Noise seed (same seed, same noise)",
            "type": "value",
            "param_info": {
                'dtype': int,
                'unit': '',
                'latex_name': '\\text{seed}',
                'default': 0
            }
        },
    }


def add_noise_formula(values: dict, config: dict) -> str:
    """The latex line under the section: the model the ticked boxes describe, with a and b."""
    return noise.formula(values)


class AddNoiseSection(BaseSectionQGroup):
    """Shared section for adding noise to the measurement."""

    title = "Add noise to measurement"
    params_ui_dict = ADD_NOISE_PARAMETERS_UI
    toml_section_key = 'add-noise'
    formula = staticmethod(add_noise_formula)


if __name__=="__main__":  # test
    import sys
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory

    import settings as settings
    from fileio.cache import make_update_cache
    from fileio import load_or_create_toml

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    # a throwaway TOML so the section's cache-sync path is exercised without touching
    # any real app cache:
    default_config = {'add-noise': {'poisson_noise': False, 'gaussian_noise': False}}
    tmp_toml = Path(tempfile.mkdtemp()) / "demo_cache.toml"

    section = AddNoiseSection(
        None,
        make_update_cache(tmp_toml, default_config),
        load_or_create_toml,
        config_path=tmp_toml,
    )
    section.resize(460, 200)
    section.show()

    sys.exit(app.exec_())
