class DisplayWindowManager:
    _windows = []
    @classmethod
    def add(cls, window):
        cls._windows.append(window)
    @classmethod
    def remove(cls, window):
        cls._windows.remove(window)

    @classmethod
    def close_all(cls):
        """Fermer toutes les fenêtres ouvertes"""
        for window in cls._windows[:]:  # Copie de la liste
            window.close()
        cls._windows.clear()
    @classmethod
    def get_count(cls):
        return len(cls._windows)