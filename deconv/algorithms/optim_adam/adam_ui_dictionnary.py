"""
Spécification des paramètres UI pour ADAM (déconvolution 2D).

Sous-ensemble simplifié de algorithms/optim_adam/adam_ui_dictionnary.py de matirf :
on retire 'delta' (anisotropie MA-TIRF) et 'rho' (SHV, sparse hessian variation).
"""

from ..utils import REGULARIZATION_LIST


ADAM_UI_PARAMETERS = {
    "max_iter": {
        "title": "Maximum iteration number",
        "type": "value",
        "param_info": {
            'dtype': int,
            'unit': '',
            'latex_name': '\\text{max\\_iter}',
            'default': 500
        }
    },
    "lr": {
        "title": "Learning rate",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': '\\text{lr}',
            'default': 0.05
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
    "lambda_reg": {
        "title": "Regularization coefficient",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': '\\lambda_{reg}',
            'default': 0.05
        }
    },
}
