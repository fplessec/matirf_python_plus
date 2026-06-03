from dataclasses import dataclass
from typing import Optional

from base.base_result import BaseResult


@dataclass
class ReconstructionResult(BaseResult):
    delta: Optional[float] = None
