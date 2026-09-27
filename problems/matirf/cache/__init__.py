from fileio.cache import make_update_cache
from problems.matirf import DEFAULT_MATIRF_CONFIG, MATIRF_CONFIG_PATH

update_cache = make_update_cache(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
