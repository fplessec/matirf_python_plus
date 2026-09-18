from common.gui.specializable.display_window import BaseDisplayWindowManager


class DisplayWindowManager(BaseDisplayWindowManager):

    @classmethod
    def _create_window(cls, pipeline):
        from matirf.gui.display_window import DisplayWindow
        return DisplayWindow(pipeline)
