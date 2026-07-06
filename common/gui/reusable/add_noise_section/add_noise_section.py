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
