from PyQt5.QtWidgets import QApplication


def close_active_window():
    window = QApplication.activeWindow()
    if window is not None:
        window.close()