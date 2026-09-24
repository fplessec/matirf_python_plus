import tomli_w
import tomli

from os.path import isfile
from pathlib import Path


def load_or_create_toml(filepath, default_config=None) -> dict:
    """Loads a .toml file. If it doesn't exist, creates it with default_config as initial content."""
    if not isfile(filepath):
        if default_config is None:
            default_config = {}
        ## create the parent directory first: some cache dirs (e.g. synthetic/cache/) are not
        ## tracked, so they are absent on a fresh checkout and the write would fail otherwise.
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as tomlfile:  # create default config is not isfile(filepath)
            tomli_w.dump(default_config, tomlfile)
    with open(filepath, "rb") as tomlfile:
        return tomli.load(tomlfile)


def save_toml(dictionary: dict, filepath) -> None:
    """Saves a python dictionary in a .toml file (name and located after filepath)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as tomlfile:
        tomli_w.dump(dictionary, tomlfile)
