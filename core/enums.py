"""
The lifecycle of a reconstruction run.

`DataMode` used to live here too, as a second, subtly different copy of the one in
`core/problem.py` — same string values, but `from_config` took the raw string rather than
the config. The two were interchangeable by value and NOT by call, which is exactly the
kind of duplicate that breaks something far away and late. There is now one `DataMode`,
declared next to `InverseProblem` where it belongs, and exported from `core`.
"""

from enum import Enum


class PipelineState(str, Enum):
    """
    Where a run is in its life.

        IDLE ──► LOADING ──► COMPUTING ──► COMPLETED
                                       └─► FAILED
                                       └─► INTERRUPTED

    >> IDLE         created, nothing started. A run rejected by validation stays here.
    >> LOADING      reading the data and building the operator.
    >> COMPUTING    the solver is iterating, on its own thread.
    >> COMPLETED    finished; the result is filled in and can be saved.
    >> FAILED       an exception escaped, or the configuration was rejected.
    >> INTERRUPTED  the user stopped it; whatever the solver had reached is kept.

    The display window listens to these transitions to start and stop the live-preview
    timer, and to enable the Save button.
    """

    IDLE = "idle"
    LOADING = "loading"
    COMPUTING = "computing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
