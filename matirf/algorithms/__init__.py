from .optim_adam import AdamAlgo
from .optim_ppxa import PpxaAlgo
from .optim_admm import AdmmAlgo
from .optim_pnp import PnpAlgo
from .mcmc_mmse_estimator import McmcAlgo


# Registry of algorithms implemented for the matirf inverse problem.
# Each class inherits name, ui_params, estimator_type from its base in common.
# Access: ALGORITHMS["ADAM"] -> class, ALGORITHMS["ADAM"].ui_params -> UI dict,
#         ALGORITHMS["ADAM"].get_ui_params() -> UI dict filtered by features.
ALGORITHMS = {cls.name: cls for cls in [
    AdamAlgo,
    PpxaAlgo,
    AdmmAlgo,
    PnpAlgo,
    McmcAlgo,
]}
