from .optim_adam import AdamAlgo
from .optim_ppxa import PpxaAlgo
from .optim_admm import AdmmAlgo
from .optim_pnp import PnpAlgo

from gui_dictionnaries.algorithms import ADAM_UI_PARAMETERS, PPXA_UI_PARAMETERS, ADMM_UI_PARAMETERS, PNP_UI_PARAMETERS


# this super mega dictionnary is imported in qgroup_algorithm_and_algo_params.py and makes the link between the ui and
# the algorithms
ALGORITHMS = {
    "ADAM": {
        'object': AdamAlgo,
        'ui_params': ADAM_UI_PARAMETERS
    },
    "PPXA": {
        'object': PpxaAlgo,
        'ui_params': PPXA_UI_PARAMETERS
    },
    "ADMM": {
        'object': AdmmAlgo,
        'ui_params': ADMM_UI_PARAMETERS
    },
    "PNP": {
        'object': PnpAlgo,
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