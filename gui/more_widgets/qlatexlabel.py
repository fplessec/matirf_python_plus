from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPalette
import matplotlib.pyplot as plt

from io import BytesIO
import re


def format_scientific_notation(text):
    """
    Detects and convert scientific notation instruction (text: str) into its latex interpretable version.
    For example: "3.67e-07" -> "3.67 \\times 10^{-7}"
    """
    pattern = r'([-+]?\d*\.?\d+)[eE]([-+]?\d+)'
    def replace_scientific(match):
        number = float(match.group(1))
        exponent = int(match.group(2))
        if '.' in str(number):
            base = f"{number:.2f}".rstrip('0').rstrip('.')
        else:
            base = f"{number}"
        exp = str(exponent).lstrip('0')
        if exp == '': exp = '0'
        # construct the latex string :
        if exp == '0':
            return base
        else:
            return f"{base} \\times 10^{{{exp}}}"
    return re.sub(pattern, replace_scientific, text)


def render_latex(formula, fontsize=12, color='w', dpi=100):
    """
    Using a latex formulation (formula: str), generates a png image (with matplotlib) to render the latex formulation,
    and that can be used for a QLatexLabel.
    """
    formatted_formula = format_scientific_notation(formula)
    fig = plt.figure(figsize=(0.01, 0.01))  # minimum size
    ax = fig.add_axes([0, 0, 1, 1])
    text = ax.text(0.5, 0.5, " " if formatted_formula in [None, ''] else f"${formatted_formula}$",
                   fontsize=fontsize,
                   color=color,
                   ha='center',
                   va='center')
    ax.set_axis_off()
    bbox = text.get_window_extent(fig.canvas.get_renderer())
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    buf = BytesIO()
    plt.savefig(buf,
                format="png",
                bbox_inches=bbox_inches,
                pad_inches=0.02,  # minimum padding
                transparent=True,
                dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


class QLatexLabel(QWidget):
    """
    A QWidget that can read latex formulations in order to display it.
    Can be updated with the method update_latex(...).
    """
    def __init__(self, formula, fontsize=12, dpi=100):
        super().__init__()
        self.fontsize = fontsize
        self.color = self.palette().color(QPalette.WindowText).name()  # the color depends on the QPalette
        self.dpi = dpi
        self.setup_ui(formula)

    def setup_ui(self, formula):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # no margin
        layout.setSpacing(0)
        #
        self.label = QLabel()
        self.label.setContentsMargins(0, 0, 0, 0)  # no margin
        self.setSizePolicy(self.label.sizePolicy())
        self.update_latex(formula)
        #
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_latex(self, latex_formula):
        buf = render_latex(latex_formula, fontsize=self.fontsize, color=self.color, dpi=self.dpi)
        # we use a QPixmap to render the latex formula image
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        self.label.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())


if __name__=="__main__":  # test
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QGroupBox, QPushButton, QStyleFactory

    import settings

    class LatexTestWidget(QGroupBox):
        def __init__(self, title='title'):
            super().__init__(title=title)
            self.setup_ui()

        def setup_ui(self):
            layout = QVBoxLayout()
            self.latex = QLatexLabel("3.67e-07 + x^2")
            btn = QPushButton("update latex")
            btn.clicked.connect(self.update_formula)
            layout.addWidget(self.latex)
            layout.addWidget(btn)
            self.setLayout(layout)

        def update_formula(self):
            import numpy as np
            val = np.random.rand() * 1e-5
            self.latex.update_latex(f"{val} + x^2")


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))

    window = LatexTestWidget(title="test of object: QLatexLabel")
    window.resize(300, 150)
    window.show()

    sys.exit(app.exec_())