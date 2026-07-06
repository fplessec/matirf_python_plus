"""
BASE METRICS LAYER (Metric primitives)

This module defines the lowest-level building blocks used to construct
quality metrics for inverse problem reconstruction evaluation.

---------------------------------------------------------------------
1. Metric
---------------------------------------------------------------------
Abstract base class for all quality metrics.

It is responsible for:
    - declaring the metric name and required features
    - computing a scalar value from (f, f_true, features)
    - adapting behavior based on problem features

This is the atomic metric unit of the system.

---------------------------------------------------------------------
2. utils
---------------------------------------------------------------------
Shared mathematical utilities used by multiple metrics.

Provides:
    - optimal_scale: computes the best scalar alignment
    - align_scale: aligns f to f_true by scale
    - to_numpy: tensor to numpy conversion

---------------------------------------------------------------------

This module is framework-agnostic with respect to inverse problems:
it only defines generic metric primitives.
"""

from .metric import Metric
from .utils import optimal_scale, align_scale, to_numpy
