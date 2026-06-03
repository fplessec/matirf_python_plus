from algorithms.denoisers import DENOISER_LIST


MCMC_UI_PARAMETERS = {
    "max_iter": {
        "title": "Number of MCMC iterations (T)",
        "type": "value",
        "param_info": {
            "dtype": int,
            'unit': '',
            'latex_name': "T",
            'default': 200
        }
    },
    "step_size": {
        "title": "Gradient step size",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': "\\tau",
            'default': 0.5
        }
    },
    "beta": {
        "title": "Temperature (acceptance selectivity)",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': "\\beta",
            'default': 0.01
        }
    },
    "sigma": {
        "title": "Noise standard deviation",
        "type": "value",
        "param_info": {
            "dtype": float,
            'unit': '',
            'latex_name': '\\sigma',
            'default': 0.05
        }
    },
    "K": {
        "title": "Log every K iterations",
        "type": "value",
        "param_info": {
            'dtype': int,
            'unit': '',
            'latex_name': 'K',
            'default': 10
        }
    },
    "denoiser": {
        "title": "Denoiser (implicit prior)",
        "type": "option",
        "param_info": {
            'options_list': DENOISER_LIST
        }
    },
}
