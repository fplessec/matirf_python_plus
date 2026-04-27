from ..denoisers import DENOISER_LIST


MCMC_UI_PARAMETERS = {
        "max_iter": {
            "title": "Number of MCMC iteration",
            "type": "value",
            "param_info": {
                "dtype": int,
                'unit': '',
                'latex_name': "\\text{max_iter}",
                'default': 100
            }
        },
        "mu": {
            "title": "Mu coefficient",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': "\mu",
                'default': 0.5
            }
        },
        "beta": {
            "title": "Beta coefficient",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': "\\beta",
                'default': 0.5
            }
        },
        "sigma": {
            "title": "Noise Standard Deviation",
            "type": "value",
            "param_info": {
                "dtype": float,
                'unit': '',
                'latex_name': '\sigma',
                'default': 0.1
            }
        },
        "EPS": {
            "title": "Stopping criterion",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\epsilon',
                'default': 1e-8
            }
        },
        "denoiser": {
            "title": "Denoiser",
            "type": "option",
            "param_info": {
                'options_list': DENOISER_LIST
            }
        }
    }