MEASUREMENT_PARAMETERS_UI = {
    "angles_deg": {
        "title": "Incident angles (deg)",
        "type": "list",
        "param_info": {
            'dtype': float,
            'hint': "1 angle per line, respecting the stack order",
        }
    },
    "n_glass": {
        "title": "Optical index of the coverslip",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': 'n_\\text{glass}',
            'default': None
        }
    },
    "n_medium": {
        "title": "Optical index of the medium",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': 'n_\\text{medium}',
            'default': None
        }
    },
    "n_oil": {
        "title": "Optical index of the immersion oil",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': 'n_\\text{oil}',
            'default': None
        }
    },
    "numerical_aperture": {
        "title": "Numerical aperture of the objective",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '',
            'latex_name': '\\text{NA}',
            'default': None
        }
    },
    "wavelength_nm": {
        "title": "Wavelength of the excitation (nm)",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': 'nm',
            'latex_name': '\lambda',
            'default': None
        }
    },
    "beam_divergence_deg": {
        "title": "Divergence of the beam (deg)",
        "type": "value",
        "param_info": {
            'dtype': float,
            'unit': '°',
            'latex_name': '\Omega',
            'default': None
        }
    }
}
