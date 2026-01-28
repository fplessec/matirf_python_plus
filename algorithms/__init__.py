from .admm_optim import AdmmOptimization
from .pytorch_optim import PytorchOptimization
from .pnp_optim import PnpOptimization
from gui_dictionnaries.algorithms import *


ALGORITHMS = {
    "Pytorch": {
        'object': PytorchOptimization,
        'ui_params': PYTORCH_UI_PARAMETERS
    },
    "ADMM": {
        'object': AdmmOptimization,
        'ui_params': ADMM_UI_PARAMETERS
    },
    "Plug and Play": {
        'object': PnpOptimization,
        'ui_params': PNP_UI_PARAMETERS
    }
}



"""
le super mega dict:
ALGORITHMS = {
    ALGORITHM_NAME: {
        "object": ALGORITHM_OBJECT,
        "ui_params": ALGORITHM_UI_PARAMETERS_DICT
    },
    ...
}
"""