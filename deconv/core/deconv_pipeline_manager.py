from base.abstract_pipeline import BasePipelineManager
from .deconv_pipeline import DeconvPipeline


class DeconvPipelineManager(BasePipelineManager):
    _pipeline_class = DeconvPipeline
    _pipelines = []
