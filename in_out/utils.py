from os.path import isfile


def _fetch(filepath):
    """Fetch a given data file from the local data dir."""
    if isfile(filepath):
        return filepath
    raise FileExistsError("Cannot find the file:", filepath)