import json

from .utils import _fetch


def load_json(filepath) -> dict:
    """Loads a .json file (located at filepath) and returns a python dictionary."""
    with open(_fetch(filepath), "r") as jsonfile:
        dictionary = json.load(jsonfile)
    return dictionary


def save_json(dictionary: dict, filepath):
    """Saves a python dictionary in a .json file (named and located after filepath)."""
    with open(filepath, 'w') as jsonfile:
        json.dump(dictionary, jsonfile, indent=4)
