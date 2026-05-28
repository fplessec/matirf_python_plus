"""
Registre global des DeconvPipeline en cours d'exécution.

Même rôle que core/matirf_pipeline_manager.py de matirf : permet à un point
central (ex: la fenêtre principale GUI) de stopper toutes les déconvolutions
en cours quand l'utilisateur ferme l'application.
"""

from .deconv_pipeline import DeconvPipeline


class DeconvPipelineManager:
    _pipelines = []

    @classmethod
    def create(cls, config, callbacks=None):
        pipeline = DeconvPipeline(config, callbacks)
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
