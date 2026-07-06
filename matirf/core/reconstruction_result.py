from dataclasses import dataclass
from typing import Optional

from common.core.base_result import BaseResult


@dataclass
class ReconstructionResult(BaseResult):
    delta: Optional[float] = None
