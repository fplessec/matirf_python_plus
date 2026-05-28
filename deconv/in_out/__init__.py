"""
Utility package for handling file input/output operations specific to the deconvolution subproject.

This module centralizes functions for reading and writing common file formats specific to the deconvolution problem
(PNG, specific TOML for specific cache), and defines key subproject paths that can be used throughout the rest of the
code.
"""


from pathlib import Path
from os.path import dirname

from in_out import (
    load_json, save_json,
    load_txt, save_txt,
    load_csv, save_csv,
)
from .png_file import load_png, save_png
from .toml_file import load_or_create_toml, save_toml


DECONV_DIR = Path(dirname(dirname(__file__)))
DECONV_CACHE_DIR = DECONV_DIR / 'cache'
DECONV_CONFIG_PATH = DECONV_CACHE_DIR / 'config.toml'
DECONV_DATA_DIR = DECONV_DIR / 'data'
DECONV_MEASUREMENTS_DIR = DECONV_DATA_DIR / 'measurements'
DECONV_RESULTS_DIR = DECONV_DATA_DIR / 'results'

