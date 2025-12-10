from pathlib import Path
from os.path import dirname

from .tif_file import load_tif, save_tif
from .json_file import load_json, save_json
from .toml_file import load_or_create_toml, save_toml
from .txt_file import load_txt, save_txt


PROJECT_DIR = Path(dirname(dirname(__file__)))
CACHE_DIR = PROJECT_DIR / "cache"
CACHED_RESULT = CACHE_DIR / "f.tif"
CONFIG_PATH = CACHE_DIR / "config.toml"
DATA_DIR = PROJECT_DIR / "data"
MEASUREMENTS_DIR = DATA_DIR / 'measurements'
RESULTS_DIR = DATA_DIR / 'results'
PYTHON_PATH = PROJECT_DIR / 'env' / 'bin' / 'python'
MAIN_PATH = PROJECT_DIR / 'main.py'
