from .matirf_pipeline import MaTirfPipeline


class PipelineManager:
    _pipelines = []

    @classmethod
    def create(cls, config, callbacks=None):
        pipeline = MaTirfPipeline(config, callbacks)
        cls._pipelines.append(pipeline)
        return pipeline

    @classmethod
    def remove(cls, pipeline):
        if pipeline in cls._pipelines:
            cls._pipelines.remove(pipeline)

    @classmethod
    def stop_all(cls):
        for p in cls._pipelines[:]:
            p.stop()
        cls._pipelines.clear()

    @classmethod
    def get_all(cls):
        return cls._pipelines