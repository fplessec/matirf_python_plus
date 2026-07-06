from pathlib import Path

from common.core.features import TWO_D
DECONV_FEATURES = {TWO_D}

DECONV_DIR = Path(__file__).resolve().parent
DECONV_CACHE_DIR = DECONV_DIR / 'cache'
DECONV_CONFIG_PATH = DECONV_CACHE_DIR / 'config.toml'
DECONV_DATA_DIR = DECONV_DIR / 'data'
DECONV_MEASUREMENTS_DIR = DECONV_DATA_DIR / 'measurements'
DECONV_RESULTS_DIR = DECONV_DATA_DIR / 'results'

DEFAULT_DECONV_CONFIG = {
    "algorithm": 'None',
    "input-paths": {
        "mode": "synthetic-data",
        "png": "None",
        "json": "None",
    },
    "add-noise": {},
    "algo-params": {},
}
