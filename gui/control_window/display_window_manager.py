class DisplayWindowManager:
    """
    This is a static singleton (it is never instantiated) which is used to organize the instantiation and the
    closures of display windows, whenever the ControlWindow needs to open new windows, or when the display windows
    themselves are closed.
    """
    _windows = []  # <-a list to store the instances

    @classmethod
    def create(cls, pipeline):
        from gui.display_window import DisplayWindow
        window = DisplayWindow(pipeline)
        cls._windows.append(window)
        return window

    @classmethod
    def remove(cls, window):
        if window in cls._windows:
            cls._windows.remove(window)

    @classmethod
    def close_all(cls):
        """Close all open windows"""
        for window in cls._windows[:]:  # <-copy of the list
            window.close()
        cls._windows.clear()

    @classmethod
    def get_all(cls):
        return cls._windows