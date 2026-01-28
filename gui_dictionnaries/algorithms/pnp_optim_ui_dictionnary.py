from ..option_lists import DENOISER_LIST


PNP_UI_PARAMETERS = {
        "sigma": {
            "title": "Noise Standard Deviation",
            "type": "value",
            "param_info": {
                "dtype": float,
                'unit': '',
                'latex_name': '\sigma',
                'default': 5
            }
        },
        "iter": {
            "title": "Number of pnp iteration",
            "type": "value",
            "param_info": {
                "dtype": int,
                'unit': '',
                'latex_name': "\\text{iter}",
                'default': 5
            }
        },
        "denoiser": {
            "title": "Denoiser",
            "type": "option",
            "param_info": {
                'options_list': DENOISER_LIST
            }
        },
        "kai_zhang": {
            "title": "Use Kai Zhang Settings",
            "type": "bool",
            "param_info": {
                'default': True
            }
        },
        "lambda_kz": {
            "title": "Kai Zhang coefficient",
            "type": "value",
            "param_info": {
                "dtype": float,
                'unit': '',
                'latex_name': '\lambda_{kz}',
                'default': 0.23
            }
        },
        "forced_pos": {
            "title": "Forced positivity",
            "type": "bool",
            "param_info": {
                'default': True
            }
        },
        "var_stab": {
            "title": "Variance Stabilization",
            "type": "bool",
            "param_info": {
                'default': True
            }
        },
    }