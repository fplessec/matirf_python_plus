from PyQt5.QtCore import QObject, pyqtSignal


class PipelineQtBridge(QObject):
    """Thread-safe bridge between pipeline callbacks and the Qt main thread."""
    message = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    state_changed = pyqtSignal(object, object)
