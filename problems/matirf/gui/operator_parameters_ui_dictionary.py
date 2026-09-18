OPERATOR_PARAMETERS_UI = {
        "nz": {
            "title": "Number of cuts on z",
            "type": "value",
            "param_info": {
                'dtype': int,
                'unit': '',
                'latex_name': 'n_z',
                'default': None
            }
        },
        "z0": {
            "title": "Smallest depth",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': 'nm',
                'latex_name': 'z_0',
                'default': None
            }
        },
        "zN": {
            "title": "Largest depth",
            "type": "value",
            "param_info": {
                'dtype': float,
                'unit': 'nm',
                'latex_name': 'z_N',
                'default': None
            }
        },
        "normalize": {
            "title": "Normalize Operator",
            "type": "bool",
            "param_info": {
                'default': False
            }
        }
    }