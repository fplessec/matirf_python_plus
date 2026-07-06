"""
Non-modal dialog that displays a spectrum of singular values (or Fourier
magnitudes) and lets the user pick a cutoff.  lambda_rr = s_k².

Used by both MA-TIRF (SVD of H) and deconvolution (FFT spectrum of PSF).
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox
from PyQt5.QtCore import Qt


class SingularValuePickerDialog(QDialog):

    def __init__(self, parent, *, title, info_text, spectrum_label_prefix,
                 S, n_show=None, footer_text=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._S = S
        n_show = n_show or len(S)

        layout = QVBoxLayout(self)

        info_label = QLabel(info_text)
        layout.addWidget(info_label)

        spectrum_lines = []
        for i in range(n_show):
            spectrum_lines.append(
                f"  {spectrum_label_prefix}_{i+1} = {S[i]:.6e}"
                f"    {spectrum_label_prefix}_{i+1}² = {S[i]**2:.6e}"
            )
        if n_show < len(S):
            spectrum_lines.append(f"  ... ({len(S)} total values)")
        spectrum_label = QLabel(
            f"Spectrum (top {n_show}):\n" + "\n".join(spectrum_lines)
        )
        layout.addWidget(spectrum_label)

        explanation = QLabel(
            "Choose the cutoff: lambda_rr = value².\n"
            "Components with value² >> lambda_rr are inverted,\n"
            "components with value² << lambda_rr are suppressed."
        )
        layout.addWidget(explanation)

        combo_label = QLabel("Cutoff value:")
        layout.addWidget(combo_label)

        self._combo = QComboBox()
        for i in range(n_show):
            self._combo.addItem(
                f"{spectrum_label_prefix}_{i+1}² = {S[i]**2:.6e}"
                f"   ({spectrum_label_prefix}_{i+1} = {S[i]:.6f})"
            )
        self._combo.setCurrentIndex(min(1, n_show - 1))
        layout.addWidget(self._combo)

        self._preview_label = QLabel()
        layout.addWidget(self._preview_label)

        self._combo.currentIndexChanged.connect(self._update_preview)
        self._update_preview()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_preview(self):
        idx = self._combo.currentIndex()
        val = (self._S[idx] ** 2).item()
        self._preview_label.setText(f"lambda_rr = {val:.6f}")

    def selected_lambda_rr(self):
        idx = self._combo.currentIndex()
        return round((self._S[idx] ** 2).item(), 6)
