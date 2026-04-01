import torch

from .custom_palette import dark_palette, light_palette


device = 'cpu'
dtype = torch.float32

app_style = "fusion"
dark_style = True

class FontSize:
    SMALL = 11
    NORMAL = 14
    BIG = 18

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


# control window settings :
width_cw = 1100
height_cw = 700
# display window settings :
width_dw = 1200
height_dw = 800

# the integrals in the computation of H are approximated by a sum of finite elements, the number of elements is
# 'precision' ; this number impacts the computation time when we calculate each element of H but doesn't impact
# the overall computation time. 100 is way enough to compute precisely the integrals and does'nt take much time.
precision = 100  # same value as in the work of Jérôme Boulanger from https://doi.org/10.1073/pnas.1414106111


normalization = 1  # normalization type for preprocessing the data file