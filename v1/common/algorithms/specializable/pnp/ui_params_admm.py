from common.denoisers import DENOISER_LIST, ANISOTROPIC_DENOISERS
from common.core.features import ANISOTROPIC


PNP_ADMM_UI_PARAMETERS = {
    "iter": {
        "title": "Number of ADMM-PnP iterations",
        "type": "value",
        "param_info": {
            "dtype": int,
            'unit': '',
            'latex_name': "\\text{iter}",
            'default': 20
        }
    },
    "rho": {
        "title": "Penalty coefficient",
        "type": "value",
        "param_info": {
            "dtype": float,
            'unit': '',
            'latex_name': '\\rho',
            'default': 1.0
        }
    },
    "sigma": {
        "title": "Denoiser Strength (0-255 scale)",
        "type": "value",
        "param_info": {
            "dtype": float,
            'unit': '',
            'latex_name': '\\sigma',
            'default': 15
        }
    },
    "denoiser": {
        "title": "Denoiser",
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
    "forced_pos": {
        "title": "Forced positivity",
        "type": "bool",
        "param_info": {
            'default': True
        }
    },
}
