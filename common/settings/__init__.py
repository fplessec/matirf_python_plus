"""
Application settings — singleton instance backed by settings.toml.

Usage::

    import common.settings as settings

    settings.device        # "cpu" / "cuda" / "mps"
    settings.dtype         # torch.float32 / torch.float64
    settings.dark_style    # True / False
    settings.FontSize.SMALL / .NORMAL / .BIG

The module-level ``__getattr__`` delegates to the singleton so that
``settings.device`` always returns the live value.
"""

import torch

from .custom_palette import dark_palette, light_palette  # noqa: F401
from .settings import Settings, _DEFAULTS  # noqa: F401

_settings = Settings()

_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class _FontSize:
    @property
    def SMALL(self):
        return _settings.get("font_size_small")

    @property
    def NORMAL(self):
        return _settings.get("font_size_normal")

    @property
    def BIG(self):
        return _settings.get("font_size_big")


FontSize = _FontSize()


def __getattr__(name):
    if name == "dtype":
        return _DTYPE_MAP[_settings.get("dtype")]
    try:
        return _settings.get(name)
    except KeyError:
        raise AttributeError(f"module 'common.settings' has no attribute {name!r}")
