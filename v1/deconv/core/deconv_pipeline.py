"""Deconvolution pipeline — subclass of BasePipeline with 7 class attributes."""

from deconv import DECONV_FEATURES
from deconv.algorithms import DECONV_ALGORITHMS
from common.core.base_pipeline import BasePipeline
from common.in_out import save_png, load_png
from .deconv_result import DeconvResult
from .pipeline_operations import DeconvOperations


class DeconvPipeline(BasePipeline):

    RESULT_CLASS = DeconvResult
    PROBLEM_FEATURES = DECONV_FEATURES
    ALGORITHM_REGISTRY = DECONV_ALGORITHMS
    PIPELINE_OPERATIONS = DeconvOperations
    SAVE_IMAGE = save_png
    LOAD_IMAGE = load_png
    IMAGE_EXTENSION = 'png'
