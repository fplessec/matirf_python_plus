"""
Base Dataclass encapsulating the outputs common to all
inverse problem-solving pipelines (MATIRF, deconv, etc.).

Subclasses add fields specific to their domain
(e.g., 'delta' for MA-TIRF's ReconstructionResult).
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaseResult:
    # always present after a successful run:
    f: Optional[torch.Tensor] = None
    H: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    # present in SYNTHETIC mode only:
    f_true: Optional[torch.Tensor] = None
    metrics: Optional[dict] = None
    diff: Optional[torch.Tensor] = None
    # execution metadata:
    messages: str = ""

    def is_complete(self) -> bool:
        return self.f is not None

    def has_synthetic_truth(self) -> bool:
        return self.f_true is not None
