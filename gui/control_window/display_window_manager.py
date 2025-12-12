class DisplayWindowManager:
    """
    This is a static singleton (it is never instantiated) which is used to organize the instantiation and the
    closures of display windows, whenever the ControlWindow needs to open new windows, or when the display windows
    themselves are closed.
    """
    _windows = []  # <-a list to store the instances
    @classmethod
    def add(cls, window):
        cls._windows.append(window)
    @classmethod
    def remove(cls, window):
        cls._windows.remove(window)

    @classmethod
    def close_all(cls):
        """Close all open windows"""
        for window in cls._windows[:]:  # <-copy of the list
            window.close()
        cls._windows.clear()
    @classmethod
    def get_count(cls):
        return len(cls._windows)