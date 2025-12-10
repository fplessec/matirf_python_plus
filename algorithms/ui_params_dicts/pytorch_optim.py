from .option_lists import REGULARIZATION_LIST

PYTORCH_UI_PARAMETERS = {
        "max_iter": {
            "title": "Maximum iteration number",
            "type": "value",
            "param_info": {
                'dtype': int,
                'unit': '',
                'latex_name': '\\text{max\\_iter}',
                'default': 1000
            }
        },
        "lr": {
            "title": "Learning rate",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\text{lr}',
                'default': 0.01
            }
        },
        "K": {
            "title": "K check",
            "type": "value",
            "param_info": {
                'dtype': int,
                'unit': '',
                'latex_name': 'K',
                'default': 10
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
        "reg": {
            "title": "Regularization",
            "type": "option",
            "param_info": {
                'options_list': REGULARIZATION_LIST
            }
        },
        "coeff": {
            "title": "Regularization coefficient",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\lambda_{reg}',
                'default': 0.10
            }
        },
        "rho": {
            "title": "Sparcity coefficient for SHV",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\rho',
                'default': 0.6
            }
        },
        "forced_pos": {
            "title": "Forced positivity",
            "type": "bool",
            "param_info": {
                'default': True
            }
        },
        "gamma": {
            "title": "Ridge regression coefficient",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\gamma',
                'default': 10000.0
            }
        },
        "random_init": {
            "title": "Random initialisation",
            "type": "bool",
            "param_info": {
                'default': False
            }
        }
    }