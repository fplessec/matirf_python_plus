from .optim_adam import AdamAlgo, ADAM_UI_PARAMETERS
from .mcmc_mmse_estimator import McmcAlgo, MCMC_UI_PARAMETERS


ALGORITHMS = {
    "ADAM": {
        'object': AdamAlgo,
        'ui_params': ADAM_UI_PARAMETERS
    },
    "MCMC": {
        'object': McmcAlgo,
        'ui_params': MCMC_UI_PARAMETERS
    },
}
