import torch

device = 'cpu'
dtype = torch.float32

class FontSize:
    SMALL = 11
    NORMAL = 14
    BIG = 18

# control window settings :
width_cw = 1100
height_cw = 700
# display window settings :
width_dw = 1200
height_dw = 800


DEFAULT_CONFIG = {
    "algorithm": 'None',
    "input-paths": {
        "mode": "real-data",
        "tif": "None",
        "json": "None"
        },
    "add-noise": {},
    "oper-params": {},
    "algo-params": {}
}