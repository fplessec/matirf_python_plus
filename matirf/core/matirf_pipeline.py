"""MA-TIRF pipeline — subclass of BasePipeline with 7 class attributes."""

from matirf import MATIRF_FEATURES
from matirf.algorithms import ALGORITHMS
from common.core.base_pipeline import BasePipeline
from common.in_out import save_tif, load_tif
from .reconstruction_result import ReconstructionResult
from .pipeline_operations import MaTirfOperations


class MaTirfPipeline(BasePipeline):

    RESULT_CLASS = ReconstructionResult
    PROBLEM_FEATURES = MATIRF_FEATURES
    ALGORITHM_REGISTRY = ALGORITHMS
    PIPELINE_OPERATIONS = MaTirfOperations
    SAVE_IMAGE = save_tif
    LOAD_IMAGE = load_tif
    IMAGE_EXTENSION = 'TIF'
