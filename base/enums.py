from enum import Enum


## input data mode: real measurement or synthetic simulation:
class DataMode(Enum):
    REAL = "real-data"
    SYNTHETIC = "synthetic-data"

    ## parses the raw TOML string into a typed DataMode:
    @classmethod
    def from_config(cls, raw_value: str) -> "DataMode":
        for mode in cls:
            if mode.value == raw_value:
                return mode
        valid = ", ".join(repr(m.value) for m in cls)
        raise ValueError(f"Unknown data mode: {raw_value!r}. Valid values: {valid}.")


## pipeline lifecycle states (IDLE -> LOADING -> COMPUTING -> COMPLETED / INTERRUPTED / FAILED):
class PipelineState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    COMPUTING = "computing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
