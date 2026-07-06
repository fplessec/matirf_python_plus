"""
Persistent application settings with TOML storage.

The Settings class holds all configurable values with their defaults,
types, descriptions, and optional choices. Values are loaded from
settings.toml at startup and written back on every change.
"""

from collections import OrderedDict
from pathlib import Path

import tomli


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.toml"

_DEFAULTS = OrderedDict([
    ("device", {
        "value": "cpu",
        "choices": ["cpu", "cuda", "mps"],
        "description": "Compute device for torch tensors",
    }),
    ("dtype", {
        "value": "float32",
        "choices": ["float32", "float64"],
        "description": "Default tensor data type",
    }),
    ("dark_style", {
        "value": True,
        "description": "Use dark theme for the GUI",
    }),
    ("app_style", {
        "value": "fusion",
        "description": "Qt application style",
    }),
    ("font_size_small", {
        "value": 11,
        "description": "Small font size (pt)",
    }),
    ("font_size_normal", {
        "value": 14,
        "description": "Normal font size (pt)",
    }),
    ("font_size_big", {
        "value": 18,
        "description": "Big font size (pt)",
    }),
    ("width_cw", {
        "value": 1100,
        "description": "Control window width (px)",
    }),
    ("height_cw", {
        "value": 700,
        "description": "Control window height (px)",
    }),
    ("width_dw", {
        "value": 1200,
        "description": "Display window width (px)",
    }),
    ("height_dw", {
        "value": 800,
        "description": "Display window height (px)",
    }),
])


def _serialize_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


class Settings:

    def __init__(self):
        self._values = {k: v["value"] for k, v in _DEFAULTS.items()}
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self):
        if SETTINGS_PATH.exists():
            with open(SETTINGS_PATH, "rb") as f:
                data = tomli.load(f)
            for k, v in data.items():
                if k in self._values:
                    self._values[k] = self._cast(k, v)
        else:
            self._save()

    def _save(self):
        lines = ["# matirf_python_plus settings", ""]
        for key, meta in _DEFAULTS.items():
            desc = meta["description"]
            choices = meta.get("choices")
            if choices:
                desc += f"  (choices: {', '.join(str(c) for c in choices)})"
            lines.append(f"# {desc}")
            lines.append(f"{key} = {_serialize_value(self._values[key])}")
            lines.append("")
        SETTINGS_PATH.write_text("\n".join(lines))

    # ── type casting ─────────────────────────────────────────────

    @staticmethod
    def _cast(key, value):
        expected_type = type(_DEFAULTS[key]["value"])
        if expected_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        return expected_type(value)

    # ── public API ───────────────────────────────────────────────

    def get(self, key):
        if key not in self._values:
            raise KeyError(f"Unknown setting: {key!r}")
        return self._values[key]

    def set(self, key, value):
        if key not in _DEFAULTS:
            raise KeyError(f"Unknown setting: {key!r}")
        value = self._cast(key, value)
        choices = _DEFAULTS[key].get("choices")
        if choices and value not in choices:
            raise ValueError(
                f"Invalid value {value!r} for '{key}'. Choices: {', '.join(choices)}"
            )
        self._values[key] = value
        self._save()

    def reset(self, key=None):
        if key is not None:
            if key not in _DEFAULTS:
                raise KeyError(f"Unknown setting: {key!r}")
            self._values[key] = _DEFAULTS[key]["value"]
        else:
            self._values = {k: v["value"] for k, v in _DEFAULTS.items()}
        self._save()

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def default(self, key):
        return _DEFAULTS[key]["value"]

    def description(self, key):
        return _DEFAULTS[key]["description"]

    def choices(self, key):
        return _DEFAULTS[key].get("choices")

    # ── display ──────────────────────────────────────────────────

    def show(self):
        max_key = max(len(k) for k in self._values)
        max_val = max(len(str(v)) for v in self._values.values())
        header = f"  {'Setting':<{max_key}}  {'Value':<{max_val}}  {'Default':<{max_val}}  Description"
        sep = "  " + "─" * (len(header) - 2)
        lines = [
            "",
            "  matirf_python_plus — Settings",
            "",
            header,
            sep,
        ]
        for k, v in self._values.items():
            default = _DEFAULTS[k]["value"]
            desc = _DEFAULTS[k]["description"]
            marker = "* " if v != default else "  "
            lines.append(f"{marker}{k:<{max_key}}  {str(v):<{max_val}}  {str(default):<{max_val}}  {desc}")
        lines.append("")
        lines.append("  * = modified from default")
        lines.append("")
        return "\n".join(lines)

    # ── attribute access ─────────────────────────────────────────

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        values = self.__dict__.get("_values")
        if values is not None and name in values:
            return values[name]
        raise AttributeError(f"No setting named {name!r}")
