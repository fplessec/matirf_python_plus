import torch
import sys
from pathlib import Path


if sys.platform.startswith("linux"):
    os_name = "linux"
elif sys.platform == "darwin":
    os_name = "mac"
elif sys.platform.startswith("win"):
    os_name = "windows"
else:
    raise RuntimeError("Your OS is not supported. It should be Linux/MacOS/Windows.")

fiji_path = None
if sys.platform.startswith("linux"):
    fiji_path = Path("/opt/Fiji.app/ImageJ-linux64")
elif sys.platform == "darwin":
    fiji_path = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx")
elif sys.platform.startswith("win"):
    fiji_path = Path(r"C:\Fiji.app\ImageJ-win64.exe")
if not fiji_path.exists() or fiji_path is None:
    raise FileNotFoundError(f"Fiji cant be found : {fiji_path}")
print(fiji_path)

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


# the integrals in the computation of H are approximated by a sum of finite elements, the number of elements is
# 'precision' ; this number impacts the computation time when we calculate each element of H but doesn't impact
# the overall computation time. 100 is way enough to compute precisely the integrals and doesnt take much time.
precision = 100