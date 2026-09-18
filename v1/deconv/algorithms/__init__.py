from .optim_adam import AdamAlgo
from .mcmc_mmse_estimator import McmcAlgo


# Registry of algorithms implemented for the deconv inverse problem.
DECONV_ALGORITHMS = {cls.name: cls for cls in [
    AdamAlgo,
    McmcAlgo,
]}
