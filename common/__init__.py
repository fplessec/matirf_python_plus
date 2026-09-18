"""
common — shared utilities that belong to no particular layer.

What remains here after the v2 rewrite is genuinely cross-cutting: file I/O, application
settings, the config cache, the denoisers, the quality metrics, and the Qt widget library.

What LEFT is just as informative. The algorithm stack moved to `solvers/`, the pipeline to
`pipeline.py`, and each problem's physics to `problems/<name>/`. `common` is no longer the
place where the framework lives — it is the place where the helpers live.
"""

from .core import DataMode, PipelineState
from .utils import get_variables_from_dict

__all__ = ["DataMode", "PipelineState", "get_variables_from_dict"]
