"""
File dialog helpers that open directly at the target directory.

Uses Qt's non-native dialog (DontUseNativeDialog) to avoid the macOS Finder
breadcrumb issue where the path bar shows the root on the right and the user
must scroll horizontally to reach the target directory.

Drop-in replacements for QFileDialog static methods:
    open_file(parent, title, directory, filter) -> path or ""
    save_file(parent, title, directory, filter) -> path or ""
    open_directory(parent, title, directory)     -> path or ""
"""

from PyQt5.QtWidgets import QFileDialog


def open_file(parent, title, directory, file_filter=""):
    """Opens a file selection dialog positioned at `directory`."""
    dialog = QFileDialog(parent, title, str(directory), file_filter)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    if dialog.exec_():
        files = dialog.selectedFiles()
        return files[0] if files else ""
    return ""


def save_file(parent, title, directory, file_filter=""):
    """Opens a file save dialog positioned at `directory`."""
    dialog = QFileDialog(parent, title, str(directory), file_filter)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    if dialog.exec_():
        files = dialog.selectedFiles()
        return files[0] if files else ""
    return ""


def open_directory(parent, title, directory):
    """Opens a directory selection dialog positioned at `directory`."""
    dialog = QFileDialog(parent, title, str(directory))
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    if dialog.exec_():
        dirs = dialog.selectedFiles()
        return dirs[0] if dirs else ""
    return ""
