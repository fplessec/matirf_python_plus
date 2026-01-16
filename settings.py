import torch
import sys

from PyQt5.QtWidgets import QApplication


device = 'cpu'
dtype = torch.float32

# get the default PyQt5 color  for 'macintosh' style :
app = QApplication(sys.argv).instance()
app.setStyle("macintosh")  # <- force the style for any OS to macintosh
window_color = app.palette().color(app.palette().Window).name()  # background color of QWidget (macintosh style)
del app
# my favorite grey color:
gray_color = '#808080'

# control window settings :
width_cw = 1100
height_cw = 700
# display window settings :
width_dw = 1200
height_dw = 800

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


# the integrals in the computation of H are approximated by a sum of finite elements, the number of elements is
# 'precision' ; this number impacts the computation time when we calculate each element of H but doesn't impact
# the overall computation time. 100 is way enough to compute precisely the integrals and doesnt take much time.
precision = 100



# if sys.platform.startswith("linux"):
#     os_name = "linux"
# elif sys.platform == "darwin":
#     os_name = "mac"
# elif sys.platform.startswith("win"):
#     os_name = "windows"
# else:
#     raise RuntimeError("Your OS is not supported. It should be Linux/MacOS/Windows.")