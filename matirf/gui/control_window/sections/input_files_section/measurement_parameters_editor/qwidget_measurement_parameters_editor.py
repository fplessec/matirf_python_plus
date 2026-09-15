"""
MA-TIRF measurement-parameters editor.

It is the generic JsonParametersEditor plus ONE problem-specific field: the list of
incident angles ('angles_deg'), entered as free text (one angle per line) rather than as a
SimpleParameterWidget. Everything else (Create/Modify, save, error handling, closeEvent)
comes from the base class.
"""

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit

from common.gui.reusable import JsonParametersEditor
from common.gui.widgets import QTextEditTab2Switch
from matirf import MATIRF_MEASUREMENTS_DIR
import common.settings as settings

from .measurement_parameters_ui_dictionary import MEASUREMENT_PARAMETERS_UI


class MeasurementParametersEditor(JsonParametersEditor):

    def __init__(self, parent):
        super().__init__(
            parent,
            params_ui_dict=MEASUREMENT_PARAMETERS_UI,
            measurements_dir=MATIRF_MEASUREMENTS_DIR,
            title_noun="Parameters",
            width=700,
        )

    # ── the 'angles_deg' extra field ────────────────────────────────────────

    def build_extra_widgets(self):
        return [self._create_angle_widget()]

    def collect_extra_parameters(self):
        """Parse the free-text incident angles into a list of floats (one per non-empty line)."""
        try:
            angles = [float(angle.strip(',')) for angle
                      in self.findChild(QTextEdit).toPlainText().splitlines() if angle.strip()]
            return {"angles_deg": angles}, ""
        except ValueError as e:
            return {}, (f"- {type(e).__name__}: {e}.\n"
                        "Please check any characters that would cause an anomaly in the incident angles "
                        "(for example a ',' instead of a '.').")

    def _create_angle_widget(self, fontsize=settings.FontSize.NORMAL):
        layout = QVBoxLayout()
        font = QFont()
        font.setPointSize(fontsize)
        layout.addLayout(self._create_first_line(qfont=font))
        input_box_widget = QTextEditTab2Switch(parent=self)
        if self.is_modify:
            input_box_widget.setPlainText("\n".join(str(a) for a in self.json_file['angles_deg']))
        input_box_widget.setFont(font)
        layout.addWidget(input_box_widget)
        container = QWidget()
        container.setLayout(layout)
        return container

    def _create_first_line(self, qfont):
        first_line = QHBoxLayout()
        title_label = QLabel("Incident angles (deg)")
        title_label.setFont(qfont)
        informative_label = QLabel("1 angle per line, respecting the stack order")
        informative_label.setStyleSheet(
            f"color: gray; font-style: italic; font-size: {settings.FontSize.SMALL}pt;")
        first_line.addWidget(title_label)
        first_line.addStretch()
        first_line.addWidget(informative_label)
        return first_line
