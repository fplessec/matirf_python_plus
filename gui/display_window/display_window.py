from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QStackedWidget, QSizePolicy, QAction
)
from tomli_w import dumps

from .sections import FiguresSection, MessageSection, SyntheticTruthSection
from gui.utils import close_active_window
from gui.control_window import DisplayWindowManager
from core import PipelineManager
from in_out import RESULTS_DIR
import settings


class PipelineQtBridge(QObject):
    message = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class DisplayWindow(QMainWindow):
    def __init__(self, pipeline):
        super().__init__()

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Display Window")
        self.resize(settings.width_dw, settings.height_dw)

        self.pipeline = pipeline
        self.config = pipeline.config

        self.qt_bridge = PipelineQtBridge()

        self.qt_bridge.message.connect(self._print, Qt.QueuedConnection)
        self.qt_bridge.finished.connect(self.on_finished, Qt.QueuedConnection)
        self.qt_bridge.error.connect(self.on_error, Qt.QueuedConnection)

        self._connect_pipeline_callbacks()

        self.setup_ui()

    def initialize_from_existing_data(self):
        self.update_plot()
        self._print(self.pipeline.messages)
        self.save_btn.setEnabled(True)

        if self.switch_btn:
            self.switch_btn.setEnabled(True)

    # =========================
    # PIPELINE
    # =========================
    def _connect_pipeline_callbacks(self):
        def emit_message(msg):
            self.qt_bridge.message.emit(msg)
        def emit_finished(result):
            self.qt_bridge.finished.emit(result)
        def emit_error(err):
            self.qt_bridge.error.emit(err)
        self.pipeline.callbacks = {
            "message": emit_message,
            "finished": emit_finished,
            "error": emit_error,
        }

    # =========================
    # CALLBACKS
    # =========================
    def on_finished(self, result):
        self.update_plot()
        self.save_btn.setEnabled(True)
        self.save_action.setEnabled(True)
        if self.switch_btn:
            self.switch_btn.setEnabled(True)

    def on_error(self, err):
        self._print(f"Error: {err}")

    def _print(self, msg):
        if hasattr(self, "messages_section"):
            self.messages_section._print(msg)

    # =========================
    # UI
    # =========================
    def setup_ui(self):
        self.create_menu_bar()
        # the base objects:
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        # create the specifics layouts/sections one after another:
        main_layout.addLayout(self.create_left_column(), 2)  # 2/7 of the width
        main_layout.addLayout(self.create_right_column(), 5)  # 5/7 of the width
        # build the objects together to make the window:
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        menubar = self.menuBar()
        # file menu:
        file_menu = menubar.addMenu("File")
            # save reconstruction action:
        self.save_action = QAction("Save reconstruction", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_reconstruction)
        file_menu.addAction(self.save_action)
        # window menu:
        window_menu = menubar.addMenu("Window")
            # close active window action:
        close_action = QAction("Close this window", self)
        close_action.setShortcut("Ctrl+W")
        close_action.setShortcutContext(Qt.WidgetShortcut)
        close_action.triggered.connect(close_active_window)
        window_menu.addAction(close_action)
            # interrupt reconstruction action:
        interrupt_action = QAction("Interrupt the reconstruction", self)
        interrupt_action.setShortcut("Ctrl+Shift+C")
        interrupt_action.triggered.connect(self.pipeline.stop)
        window_menu.addAction(interrupt_action)

    def create_left_column(self):
        left_layout = QVBoxLayout()
        # config section:
        config_section = MessageSection("Config")
        config_section._print(dumps(self.config))
        # messages section
        self.messages_section = MessageSection("Messages")
        # build the section together in one column:
        left_layout.addWidget(config_section, 1)  # 1/3 of the height
        left_layout.addWidget(self.messages_section, 2)  # 2/3 of the height
        left_layout.addWidget(self.create_bottom_bar())
        return left_layout

    def create_right_column(self):
        right_layout = QVBoxLayout()
        # right column is a QStackedWidget and can switch between some sections:
        # if working with real data: only figures_section
        # if working with synth data: also a synthetic_truth_section
        self.stack = QStackedWidget()
        self.figures_section = FiguresSection(parent=self)
        self.stack.addWidget(self.figures_section)   # index = 0
        if self.pipeline.is_synthetic_data():
            self.synthetic_section = SyntheticTruthSection(self.pipeline)
            self.stack.addWidget(self.synthetic_section)   # index = 1
        else:
            self.synthetic_section = None
        right_layout.addWidget(self.stack)
        return right_layout

    def create_bottom_bar(self):
        widget = QWidget()
        bottom_bar_layout = QVBoxLayout()
        # a save button: (enable on algo finished)
        self.save_btn = QPushButton("Save reconstruction")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_reconstruction)
        self.save_btn.setSizePolicy(QSizePolicy.Fixed,      # horizontally: does not stretch
                                    QSizePolicy.Preferred)  # vertically: does stretch
        bottom_bar_layout.addWidget(self.save_btn, alignment=Qt.AlignCenter)
        bottom_bar_layout.addStretch()
        # a button to acces to synthetic_truth_section: (enable on algo finished)
        if self.pipeline.is_synthetic_data():
            self.switch_btn = QPushButton("Go To Truth →")
            self.switch_btn.setEnabled(False)
            self.switch_btn.clicked.connect(self.change_right_column)
            self.switch_btn.setSizePolicy(QSizePolicy.Fixed,      # horizontally: does not stretch
                                          QSizePolicy.Preferred)  # vertically: does stretch
            bottom_bar_layout.addWidget(self.switch_btn, alignment=Qt.AlignCenter)
        else:
            self.switch_btn = None
        widget.setLayout(bottom_bar_layout)
        return widget

    # =========================
    # DISPLAY
    # =========================
    def update_plot(self):
        self.figures_section.update_plot(self.pipeline.f, self.config)
        if self.pipeline.is_synthetic_data() and self.synthetic_section:
            self.synthetic_section.update_plot()

    def change_right_column(self):
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
            self.switch_btn.setText("← Go to Reconstruction")
        else:
            self.stack.setCurrentIndex(0)
            self.switch_btn.setText("Go To Truth →")


    def save_reconstruction(self):
        """
        Opens a dialog box to create a folder and saves the reconstruction.
        Each reconstruction is a folder that contains data and metadata, such as the configuration file.
        """
        save_dir, _ = QFileDialog.getSaveFileName(self, "Name the Save Directory",
                                                  str(RESULTS_DIR), "Folder Selection (*.*)")
        if not save_dir:
            return
        self.pipeline.save_results(save_dir)

    def closeEvent(self, event):
        """Rewrites the closeEvent to stop the algorithm correctly and communicate with the managers."""
        if self.synthetic_section:
            if self.synthetic_section.viewer_window:
                self.synthetic_section.viewer_window.close()
        if self.pipeline:
            self.pipeline.stop()
            PipelineManager.remove(self.pipeline)
        DisplayWindowManager.remove(self)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # escape is pressed
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)