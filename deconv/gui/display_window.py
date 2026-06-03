from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QStackedWidget, QSizePolicy, QAction,
)
from tomli_w import dumps

from gui.display_window.sections import MessageSection
from gui.utils import close_active_window
from .display_window_manager import DeconvDisplayWindowManager
from .figures_section import DeconvFiguresSection
from .synthetic_truth_section import DeconvSyntheticTruthSection
from deconv.core import DeconvPipelineManager
from base import PipelineState
from deconv.in_out import DECONV_RESULTS_DIR
import settings


## thread-safe bridge between pipeline callbacks and the Qt main thread:
class PipelineQtBridge(QObject):
    message = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    state_changed = pyqtSignal(object, object)


class DeconvDisplayWindow(QMainWindow):

    def __init__(self, pipeline):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Deconv Display Window")
        self.resize(settings.width_dw, settings.height_dw)
        self.pipeline = pipeline
        self.config = pipeline.config
        self.qt_bridge = PipelineQtBridge()
        self.qt_bridge.message.connect(self._print, Qt.QueuedConnection)
        self.qt_bridge.finished.connect(self._on_finished, Qt.QueuedConnection)
        self.qt_bridge.error.connect(self._on_error, Qt.QueuedConnection)
        self.qt_bridge.state_changed.connect(self._on_state_changed, Qt.QueuedConnection)
        self._connect_pipeline_callbacks()
        self._setup_ui()

    def initialize_from_existing_data(self):
        self._update_plot()
        self._print(self.pipeline.result.messages)
        self.save_btn.setEnabled(True)
        if self.switch_btn:
            self.switch_btn.setEnabled(True)

    ## wires pipeline callbacks to Qt signals:
    def _connect_pipeline_callbacks(self):
        self.pipeline.callbacks = {
            "message": lambda msg: self.qt_bridge.message.emit(msg),
            "finished": lambda result: self.qt_bridge.finished.emit(result),
            "error": lambda err: self.qt_bridge.error.emit(err),
            "state_changed": lambda old, new: self.qt_bridge.state_changed.emit(old, new),
        }

    def _on_finished(self, result):
        self._update_plot()
        self.save_btn.setEnabled(True)
        self.save_action.setEnabled(True)
        if self.switch_btn:
            self.switch_btn.setEnabled(True)

    def _on_error(self, err):
        self._print(f"Error: {err}")

    def _on_state_changed(self, old_state: PipelineState, new_state: PipelineState):
        self._print(f"[state] {old_state.value} -> {new_state.value}")

    def _print(self, msg):
        if hasattr(self, "messages_section"):
            self.messages_section._print(msg)

    ## builds the main layout: left (config + messages) and right (figures):
    def _setup_ui(self):
        self._create_menu_bar()
        central = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addLayout(self._create_left_column(), 2)
        main_layout.addLayout(self._create_right_column(), 5)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        self.save_action = QAction("Save reconstruction", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save)
        file_menu.addAction(self.save_action)
        window_menu = menubar.addMenu("Window")
        close_action = QAction("Close this window", self)
        close_action.setShortcut("Ctrl+W")
        close_action.setShortcutContext(Qt.WidgetShortcut)
        close_action.triggered.connect(close_active_window)
        window_menu.addAction(close_action)
        interrupt = QAction("Interrupt the reconstruction", self)
        interrupt.setShortcut("Ctrl+Shift+C")
        interrupt.triggered.connect(self.pipeline.stop)
        window_menu.addAction(interrupt)

    def _create_left_column(self):
        layout = QVBoxLayout()
        config_section = MessageSection("Config")
        config_section._print(dumps(self.config))
        self.messages_section = MessageSection("Messages")
        layout.addWidget(config_section, 1)
        layout.addWidget(self.messages_section, 2)
        layout.addWidget(self._create_bottom_bar())
        return layout

    def _create_right_column(self):
        layout = QVBoxLayout()
        self.stack = QStackedWidget()
        self.figures_section = DeconvFiguresSection(parent=self)
        self.stack.addWidget(self.figures_section)
        if self.pipeline.is_synthetic_data():
            self.synthetic_section = DeconvSyntheticTruthSection(self.pipeline)
            self.stack.addWidget(self.synthetic_section)
        else:
            self.synthetic_section = None
        layout.addWidget(self.stack)
        return layout

    def _create_bottom_bar(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.save_btn = QPushButton("Save reconstruction")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(self.save_btn, alignment=Qt.AlignCenter)
        layout.addStretch()
        if self.pipeline.is_synthetic_data():
            self.switch_btn = QPushButton("Go To Truth ->")
            self.switch_btn.setEnabled(False)
            self.switch_btn.clicked.connect(self._change_right_column)
            self.switch_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            layout.addWidget(self.switch_btn, alignment=Qt.AlignCenter)
        else:
            self.switch_btn = None
        widget.setLayout(layout)
        return widget

    def _update_plot(self):
        self.figures_section.update_plot(self.pipeline.result.f, self.config)
        if self.pipeline.is_synthetic_data() and self.synthetic_section:
            self.synthetic_section.update_plot()

    def _change_right_column(self):
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
            self.switch_btn.setText("<- Go to Reconstruction")
        else:
            self.stack.setCurrentIndex(0)
            self.switch_btn.setText("Go To Truth ->")

    def _save(self):
        save_dir, _ = QFileDialog.getSaveFileName(
            self, "Name the Save Directory", str(DECONV_RESULTS_DIR), "Folder Selection (*.*)")
        if not save_dir:
            return
        self.pipeline.save_results(save_dir)

    ## stops the pipeline and unregisters from managers before closing:
    def closeEvent(self, event):
        if self.synthetic_section:
            if self.synthetic_section.viewer_window:
                self.synthetic_section.viewer_window.close()
        if self.pipeline:
            self.pipeline.stop()
            DeconvPipelineManager.remove(self.pipeline)
        DeconvDisplayWindowManager.remove(self)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)
