import tomli_w
import tomli

from in_out.utils import isfile
from settings import DEFAULT_CONFIG


def load_or_create_toml(filepath) -> dict:
    """Loads a .toml file (located at filepath) and returns a python dictionary.
    If the file doesn't exist it will be created and will return an empty dictionary."""
    if not isfile(filepath):
        with open(filepath, "wb") as tomlfile:
            tomli_w.dump(DEFAULT_CONFIG, tomlfile)
    with open(filepath, "rb") as tomlfile:
        return tomli.load(tomlfile)


def save_toml(dictionary: dict, filepath):
    """Saves a python dictionary in a .toml file (name and located after filepath)."""
    with open(filepath, "wb") as tomlfile:
        tomli_w.dump(dictionary, tomlfile)
