from common.gui.base.base_section_qgroup import BaseSectionQGroup


ADD_NOISE_PARAMETERS_UI = {
        "add_noise": {
            "title": "Add noise",
            "type": "bool",
            "param_info": {
                'default': False
            }
        },
        "is_gaussian": {
            "title": "Gaussian Noise",
            "type": "bool",
            "param_info": {
                'default': True
            }
        },
        "sigma": {
            "title": "Noise standard deviation",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\sigma',
                'default': None
            }
        }
    }


class AddNoiseSection(BaseSectionQGroup):
    """Shared section for adding noise to the measurement."""

    title = "Add noise to measurement"
    params_ui_dict = ADD_NOISE_PARAMETERS_UI
    toml_section_key = 'add-noise'


if __name__=="__main__":  # test
    import sys
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory

    import common.settings as settings
    from common.cache import make_update_cache
    from common.in_out import load_or_create_toml

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    # a throwaway TOML so the section's cache-sync path is exercised without touching
    # any real app cache:
    default_config = {'add-noise': {'add_noise': False, 'is_gaussian': True, 'sigma': 'null'}}
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
