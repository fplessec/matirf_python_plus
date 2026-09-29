"""
Result — everything one reconstruction run produced.

Pure data: no behaviour beyond two convenience predicates. It is what the display window
reads, what gets written to disk, and what a benchmark compares.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Result:
    """
    The outcome of a run.

    ----------
    > Always present after a successful run :
    ----------

    >> f : torch.Tensor
        The reconstruction — the answer.

    >> g : torch.Tensor
        The measurement it was computed from, kept so the run is self-contained.

    >> messages : str
        Everything the pipeline and the solver reported, in order. Written to messages.txt,
        so a finished run explains itself without the terminal it was launched from.

    ----------
    > Present in SYNTHETIC mode only :
    ----------

    >> f_true : torch.Tensor
        The known object the measurement was simulated from.

    >> metrics : dict
        One entry per quality metric: a float for a scalar metric, or a dict
        {summary, x, y, xlabel, ylabel} for a curve metric such as FSC.

    >> diff : torch.Tensor
        f_true - alpha * f, the error image. alpha is the optimal scale factor, fitted
        first because a scale-ambiguous problem determines f only up to a constant —
        subtracting without it would show that constant rather than the error.

    >> alpha : float
        The fitted scale factor, kept so the difference image can be interpreted.

    ----------
    > Present after every successful run :
    ----------

    >> diagnosis : core.diagnostics.Diagnosis
        Whether f is a plausible reconstruction, or one of the known meaningless shapes
        (collapsed at the interface, the same image on every plane, f ~ g, f ~ a
        least-squares estimate). Its summary is also written to the log.
    """

    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    messages: str = ""

    f_true: Optional[torch.Tensor] = None
    metrics: Optional[dict] = None
    diff: Optional[torch.Tensor] = None
    alpha: Optional[float] = None
    diagnosis: Optional[object] = None

    extras: dict = field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.f is not None

    def has_synthetic_truth(self) -> bool:
        return self.f_true is not None
