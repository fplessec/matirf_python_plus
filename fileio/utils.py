"""
Shared fileio helper: locating a data file on disk (`_fetch`).
"""

from os.path import isfile


def _fetch(filepath) -> str:
    """Fetch a given data file from the local data dir."""
    if isfile(filepath):
        return filepath
    raise FileNotFoundError(f"Cannot find the file: {filepath}")