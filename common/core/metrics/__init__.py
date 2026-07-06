"""
METRICS ARCHITECTURE OVERVIEW (Inverse Problems Framework)

This package implements the quality metrics system used to evaluate
reconstructions across all inverse problem modules (matirf, deconv,
and future extensions).

Metrics are autonomous evaluation functions: they take a reconstruction f
and a ground truth f_true, and return a scalar quality measure.

The system is organized into two layers:

=====================================================================
1. base/
=====================================================================

Low-level building blocks for quality metrics.

    - Metric: abstract base class with name, requires, compute()
    - utils: optimal_scale, align_scale, to_numpy

=====================================================================
2. reusable/
=====================================================================

Concrete metric implementations and registry.

    - Universal: MSE, MAE, NMSE, Correlation, cosine similarity,
                 angular distance, PSNR, SSIM
    - 3D-only (requires={"3d"}): Sinkhorn Wasserstein, FSC
    - Scale-ambiguous only (requires={"scale_ambiguous"}): Scale_alpha

=====================================================================
Adaptive behavior
=====================================================================

Each metric receives the problem's `features` set and adapts:
    - "scale_ambiguous" in features → aligns scale before comparing
    - "3d" in features → handles volumetric data appropriately
    - "2d" in features → uses standard 2D implementations

The `requires` set filters out metrics that don't apply:
    - requires ⊆ features → metric is included
    - otherwise → metric is skipped

=====================================================================
Usage
=====================================================================

    from common.core.metrics import compute_all_metrics

    # matirf (3D, scale-ambiguous reconstruction):
    metrics = compute_all_metrics(f, f_true, features={"3d", "scale_ambiguous"}, delta=0.05)

    # deconv (2D, direct comparison):
    metrics = compute_all_metrics(f, f_true, features={"2d"})
"""

from .base import Metric, optimal_scale, align_scale
from .reusable import METRIC_REGISTRY, METRIC_LIST


def compute_all_metrics(f, f_true, features=set(), **kw):
    """
    Computes all metrics whose requirements are met by the given features.

    Args:
        f: reconstructed tensor
        f_true: ground truth tensor
        features: set of problem features (e.g. {"3d", "scale_ambiguous"})
        **kw: extra keyword arguments passed to each metric (e.g. delta)

    Returns:
        dict mapping metric name -> value, where value is:
            - a float for scalar metrics
            - a dict {"summary", "x", "y", "xlabel", "ylabel"} for curve metrics
    """
    results = {}
    f = f.detach()
    f_true = f_true.detach()
    for name, metric in METRIC_REGISTRY.items():
        if metric.requires <= features:
            try:
                results[name] = metric.compute(f, f_true, features=features, **kw)
            except Exception as e:
                results[name] = f"Error: {e}"
    return results
