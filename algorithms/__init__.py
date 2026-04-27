from .optim_adam import AdamAlgo, ADAM_UI_PARAMETERS
from .optim_ppxa import PpxaAlgo, PPXA_UI_PARAMETERS
from .optim_admm import AdmmAlgo, ADMM_UI_PARAMETERS
from .optim_pnp import PnpAlgo, PNP_UI_PARAMETERS
from .mcmc_mmse_estimator import McmcAlgo, MCMC_UI_PARAMETERS


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
    },
    "MCMC": {
        'object': McmcAlgo,
        'ui_params': MCMC_UI_PARAMETERS
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