class Metric:
    """
    Base class for quality metrics comparing a reconstruction f to a ground truth f_true.

    Each subclass provides:
        - name:        display name for the metric (used as registry key)
        - requires:    set of features needed for this metric to be meaningful
                       (e.g. {"3d"} for metrics that only make sense on volumes)
        - result_type: "scalar" or "curve"
        - compute(f, f_true, features, **kw): the metric value

    The `features` argument is passed at call time so each metric can adapt
    its behavior (e.g. align scale if "scale_ambiguous" is in features,
    compute slice-by-slice if "3d" is in features).

    The `requires` set is used for filtering: a metric is only called
    if requires is a subset of the problem's features.

    Return format:
        - scalar metrics return a float
        - curve metrics return a dict:
            {"summary": float, "x": list, "y": list,
             "xlabel": str, "ylabel": str}
    """

    name = ""
    requires = set()
    result_type = "scalar"   # "scalar" or "curve"

    def _prepare(self, f, f_true, features):
        """Handle scale alignment for scale-ambiguous problems. Returns (f_prepared, f_true)."""
        from common.core.features import SCALE_AMBIGUOUS
        from .utils import align_scale
        if SCALE_AMBIGUOUS in features:
            f, _ = align_scale(f, f_true)
        return f, f_true

    def compute(self, f, f_true, features=set(), **kw):
        """Compute the metric value. Returns a float or a curve dict."""
        raise NotImplementedError
