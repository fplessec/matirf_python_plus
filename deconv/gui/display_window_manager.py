from common.gui.specializable.display_window import BaseDisplayWindowManager


class DeconvDisplayWindowManager(BaseDisplayWindowManager):

    @classmethod
    def _create_window(cls, pipeline):
        from .display_window import DeconvDisplayWindow
        return DeconvDisplayWindow(pipeline)
