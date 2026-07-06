class BaseDisplayWindowManager:
    """
    Static singleton that manages display window instances.

    Each concrete subclass gets its own `_windows` list via __init_subclass__
    and must override `_create_window(pipeline)` to return the appropriate DisplayWindow.
    """
    _windows = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._windows = []

    @classmethod
    def _create_window(cls, pipeline):
        raise NotImplementedError

    @classmethod
    def create(cls, pipeline):
        window = cls._create_window(pipeline)
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
