from common.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS
from common.algorithms.reusable.data_fidelities import DATA_FIDELITY_LIST
from common.core.features import ANISOTROPIC


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
    "mu": {
        "title": "Mu coefficient",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': "\\mu",
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
    "denoiser": {
        "title": "Denoiser (implicit prior)",
        "type": "option",
        "param_info": {
            'options_list': DENOISER_LIST
        }
    },
    "delta": {
        "title": "Anisotropy ratio coefficient",
        "type": "value",
        "requires": {ANISOTROPIC},
        "depends_on": {"denoiser": ANISOTROPIC_DENOISERS},
        "param_info": {
            "dtype": float,
            'unit': '',
            'latex_name': '\\delta = \\frac{\\Delta z}{\\Delta xy}',
            'default': 1.0
        }
    },
    "data_fidelity": {
        "title": "Data fidelity (noise model)",
        "type": "option",
        "param_info": {
            'options_list': DATA_FIDELITY_LIST
        }
    },
}
