"""
Cache specific to the subproject deconv
"""


from deconv.in_out import DECONV_CONFIG_PATH, load_or_create_toml, save_toml


def update_cache(key_path, new_value):
    """
    Modifies a value in the deconv cached config file at a particular key,
    without modifying the value of the other keys.
    Args:
        key_path (list of str): list representing the path to the key
                                (for example: ['psf-params', 'sigma']).
        new_value: the new value to affect.
    """
    config = load_or_create_toml(DECONV_CONFIG_PATH)
    ref = config
    for key in key_path[:-1]:
        ref = ref[key]
    # if the new value is None, it will be represented by "null" in the .toml cached config file
    ref[key_path[-1]] = new_value if new_value is not None else "null"
    save_toml(config, DECONV_CONFIG_PATH)