## static singleton that manages deconv display windows:
class DeconvDisplayWindowManager:
    _windows = []

    @classmethod
    def create(cls, pipeline):
        from .display_window import DeconvDisplayWindow
        window = DeconvDisplayWindow(pipeline)
        cls._windows.append(window)
        return window

    @classmethod
    def remove(cls, window):
        if window in cls._windows:
            cls._windows.remove(window)

    @classmethod
    def close_all(cls):
        for window in cls._windows[:]:
            window.close()
        cls._windows.clear()

    @classmethod
    def get_all(cls):
        return cls._windows
