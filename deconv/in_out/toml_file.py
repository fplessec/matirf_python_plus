"""
Lecture / écriture TOML spécifique au sous-projet deconv.
Specific TOML i/o because we can to use DEFAULT_DECONV_CONFIG as a default confing.toml
     > DEFAULT_DECONV_CONFIG from deconv/settings/__init__.py is for the deconv problem
     > DEFAULT_CONFIG from settings/__init__.py is for the MA-TIRF problem
"""

import tomli
import tomli_w

from in_out.utils import isfile
from deconv.settings import DEFAULT_DECONV_CONFIG


def load_or_create_toml(filepath) -> dict:
    """
    Loads a .toml file (located at filepath) and returns a python dictionary.
    If the file doesn't exist it will be created with DEFAULT_DECONV_CONFIG
    as initial content.
    """
    if not isfile(filepath):
        with open(filepath, "wb") as tomlfile:
            tomli_w.dump(DEFAULT_DECONV_CONFIG, tomlfile)
    with open(filepath, "rb") as tomlfile:
        return tomli.load(tomlfile)


def save_toml(dictionary: dict, filepath):
    """Saves a python dictionary in a .toml file (name and located after filepath)."""
    with open(filepath, "wb") as tomlfile:
        tomli_w.dump(dictionary, tomlfile)
