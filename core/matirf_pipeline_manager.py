from base.abstract_pipeline import BasePipelineManager
from .matirf_pipeline import MaTirfPipeline


class PipelineManager(BasePipelineManager):
    _pipeline_class = MaTirfPipeline
    _pipelines = []
