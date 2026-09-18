from problems.deconv import DECONV_CONFIG_PATH, DEFAULT_DECONV_CONFIG
from fileio.cache import make_update_cache

update_cache = make_update_cache(DECONV_CONFIG_PATH, DEFAULT_DECONV_CONFIG)
