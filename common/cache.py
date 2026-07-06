from common.in_out import load_or_create_toml, save_toml


def make_update_cache(config_path, default_config):
    """
    Returns an update_cache(key_path, new_value) function bound to a specific
    config_path and default_config.
    """
    def update_cache(key_path, new_value):
        config = load_or_create_toml(config_path, default_config)
        ref = config
        for key in key_path[:-1]:
            ref = ref[key]
        ref[key_path[-1]] = new_value if new_value is not None else "null"
        save_toml(config, config_path)
    return update_cache
