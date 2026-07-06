from pathlib import Path

from common.core.features import THREE_D, SCALE_AMBIGUOUS, ANISOTROPIC
MATIRF_FEATURES = {THREE_D, SCALE_AMBIGUOUS, ANISOTROPIC}

MATIRF_DIR = Path(__file__).resolve().parent
MATIRF_CACHE_DIR = MATIRF_DIR / 'cache'
MATIRF_CONFIG_PATH = MATIRF_CACHE_DIR / 'config.toml'
MATIRF_DATA_DIR = MATIRF_DIR / 'data'
MATIRF_MEASUREMENTS_DIR = MATIRF_DATA_DIR / 'measurements'
MATIRF_RESULTS_DIR = MATIRF_DATA_DIR / 'results'

DEFAULT_MATIRF_CONFIG = {
    "algorithm": 'None',
    "input-paths": {
        "mode": "real-data",
        "tif": "None",
        "json": "None",
    },
    "add-noise": {},
    "oper-params": {},
    "algo-params": {},
}
