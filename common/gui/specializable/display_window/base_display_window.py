"""
Base display window for any inverse problem.

Provides the full UI skeleton (menu bar, config/messages, figures stack,
pipeline bridge) and delegates problem-specific parts to hooks.
"""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QStackedWidget, QSizePolicy, QAction,
)
from common.gui.file_dialog import save_file
from tomli_w import dumps

from common.gui.utils import close_active_window
from common.core.enums import PipelineState
from common.gui.specializable.display_window.pipeline_qt_bridge import PipelineQtBridge
import common.settings as settings


class BaseDisplayWindow(QMainWindow):

    def __init__(self, pipeline):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(self.window_title())
        self.resize(settings.width_dw, settings.height_dw)

        self.pipeline = pipeline
        self.config = pipeline.config

        self.qt_bridge = PipelineQtBridge()
        self.qt_bridge.message.connect(self._print, Qt.QueuedConnection)
        self.qt_bridge.finished.connect(self._on_finished, Qt.QueuedConnection)
        self.qt_bridge.error.connect(self._on_error, Qt.QueuedConnection)
        self.qt_bridge.state_changed.connect(self._on_state_changed, Qt.QueuedConnection)
        self._connect_pipeline_callbacks()

        ## timer for live preview: polls the algorithm's latest f snapshot at a fixed interval
        ## instead of relying on Qt signals (which would pile up and freeze the UI):
        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setInterval(250)  # ms
        self._live_preview_timer.timeout.connect(self._poll_live_preview)
        self._last_polled_f = None

        self._setup_ui()

    # ── hooks (to override) ──────────────────────────────────────────────

    def window_title(self) -> str:
        raise NotImplementedError

    def results_dir(self):
        raise NotImplementedError

    def pipeline_class(self):
        raise NotImplementedError

    def display_window_manager_class(self):
        raise NotImplementedError

    def create_figures_section(self):
        raise NotImplementedError

    def create_synthetic_section(self):
        return None

    def update_figures(self, f, config):
        raise NotImplementedError

    def update_synthetic(self):
        pass

    def on_close_synthetic_cleanup(self):
        pass

    # ── pipeline bridge ──────────────────────────────────────────────────

    def _connect_pipeline_callbacks(self):
        self.pipeline.on_message = lambda msg: self.qt_bridge.message.emit(msg)
        self.pipeline.on_finished = lambda result: self.qt_bridge.finished.emit(result)
        self.pipeline.on_error = lambda err: self.qt_bridge.error.emit(err)
        self.pipeline.on_state_changed = lambda old, new: self.qt_bridge.state_changed.emit(old, new)

    def _on_finished(self, result):
        self._live_preview_timer.stop()
        self._update_plot()
        self.save_btn.setEnabled(True)
        self.save_action.setEnabled(True)
        if self.switch_btn:
            self.switch_btn.setEnabled(True)

    def _on_error(self, err):
        self._live_preview_timer.stop()
        self._print(f"Error: {err}")

    def _on_state_changed(self, old_state: PipelineState, new_state: PipelineState):
        ## start the live preview timer when the algorithm starts computing:
        if new_state == PipelineState.COMPUTING:
            self._last_polled_f = None
            self._live_preview_timer.start()
        ## stop the timer when the algorithm finishes or fails:
        elif new_state in (PipelineState.COMPLETED, PipelineState.FAILED, PipelineState.INTERRUPTED):
            self._live_preview_timer.stop()

    ## polls the algorithm's latest f snapshot for live preview
    def _poll_live_preview(self):
        if self.pipeline.algorithm is None:
            return
        f = self.pipeline.algorithm._latest_f
        if f is None or f is self._last_polled_f:
            return
        self._last_polled_f = f
        if hasattr(self, 'figures_section') and self.figures_section is not None:
            self.update_figures(f, self.config)

    def _print(self, msg):
        if hasattr(self, "messages_section"):
            self.messages_section._print(msg)

    # ── public API ───────────────────────────────────────────────────────

    def initialize_from_existing_data(self):
        self._update_plot()
        self._print(self.pipeline.result.messages)
        self.save_btn.setEnabled(True)
        if self.switch_btn:
            self.switch_btn.setEnabled(True)

    # ── UI construction ──────────────────────────────────────────────────

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
        # file menu:
        file_menu = menubar.addMenu("File")
        self.save_action = QAction("Save reconstruction", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save)
        file_menu.addAction(self.save_action)
        # window menu:
        window_menu = menubar.addMenu("Window")
        close_action = QAction("Close this window", self)
        close_action.setShortcut("Ctrl+W")
        close_action.setShortcutContext(Qt.WidgetShortcut)
        close_action.triggered.connect(close_active_window)
        window_menu.addAction(close_action)
        interrupt_action = QAction("Interrupt the reconstruction", self)
        interrupt_action.setShortcut("Ctrl+Shift+C")
        interrupt_action.triggered.connect(self.pipeline.stop)
        window_menu.addAction(interrupt_action)

    def _create_left_column(self):
        from common.gui.reusable.message_section import MessageSection
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
        self.figures_section = self.create_figures_section()
        self.stack.addWidget(self.figures_section)
        if self.pipeline.is_synthetic_data:
            self.synthetic_section = self.create_synthetic_section()
            if self.synthetic_section:
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
        if self.pipeline.is_synthetic_data:
            self.switch_btn = QPushButton("Go To Truth →")
            self.switch_btn.setEnabled(False)
            self.switch_btn.clicked.connect(self._change_right_column)
            self.switch_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            layout.addWidget(self.switch_btn, alignment=Qt.AlignCenter)
        else:
            self.switch_btn = None
        widget.setLayout(layout)
        return widget

    # ── shared actions ───────────────────────────────────────────────────

    def _update_plot(self):
        self.update_figures(self.pipeline.result.f, self.config)
        if self.pipeline.is_synthetic_data and self.synthetic_section:
            self.update_synthetic()

    def _change_right_column(self):
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
            self.switch_btn.setText("← Go to Reconstruction")
        else:
            self.stack.setCurrentIndex(0)
            self.switch_btn.setText("Go To Truth →")

    def _save(self):
        save_dir = save_file(self, "Name the Save Directory", self.results_dir(), "Folder Selection (*.*)")
        if not save_dir:
            return
        self.pipeline.save_results(save_dir)

    # ── close / key events ───────────────────────────────────────────────

    def closeEvent(self, event):
        self._live_preview_timer.stop()
        self.on_close_synthetic_cleanup()
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline_class().remove(self.pipeline)
        self.display_window_manager_class().remove(self)
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            widget = self.focusWidget()
            if widget is not None:
                widget.clearFocus()
        else:
            super().keyPressEvent(event)
