from ..option_lists import PPXA_REGULARIZATION_LIST


PPXA_UI_PARAMETERS = {
        "max_iter": {
            "title": "Maximum iteration number",
            "type": "value",
            "param_info": {
                "dtype": int,
                'unit': '',
                'latex_name': "\\text{max_iter}",
                'default': 2000
            }
        },
        "lambda_relax": {
            "title": "Relaxation parameter",
            "type": "value",
            "param_info": {
                "dtype": float,
                'unit': '',
                'latex_name': '\lambda_{relax}',
                'default': 1.9
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
        "gamma": {
            "title": "Data term step size",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': '',
                'latex_name': '\\gamma',
                'default': 0.05
            }
        },
        "reg": {
            "title": "Regularization",
            "type": "option",
            "param_info": {
                'options_list': PPXA_REGULARIZATION_LIST
            }
        },
        "lambda_reg": {
            "title": "Regularization coefficient",
            "type": "value",
            "param_info": {
                "dtype": float,
                'unit': '',
                'latex_name': '\lambda_{reg}',
                'default': 0.001
            }
        },
        "delta": {
            "title": "Anisotropy ratio coefficient",
            "type": "value",
            "param_info": {
                "dtype": float,
                'unit': '',
                'latex_name': '\delta = \\frac{\Delta z}{\Delta xy}',
                'default': 0.05
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
    }