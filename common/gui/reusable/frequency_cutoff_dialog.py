"""
Non-modal dialog for choosing lambda_rr on a logarithmic scale.

Adapted to continuous spectra (e.g. Gaussian PSF) where individual
singular values are too close to pick one by one.  The slider controls
log10(lambda_rr) directly; feedback shows how many frequencies are
effectively kept or suppressed at the chosen value.
"""

import math

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDialogButtonBox,
)
from PyQt5.QtCore import Qt

import torch


_SLIDER_STEPS = 500


class FrequencyCutoffDialog(QDialog):

    def __init__(self, parent, *, title, info_text, S):
        """
        S: 1-D tensor of |H_fft| values sorted descending.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._S_sq = (S ** 2)
        self._n_total = len(S)

        sq_max = self._S_sq[0].item()
        sq_min = self._S_sq[-1].item()
        self._log_min = math.floor(math.log10(max(sq_min, 1e-20)))
        self._log_max = math.ceil(math.log10(max(sq_max, 1e-20)))

        layout = QVBoxLayout(self)

        info_label = QLabel(info_text)
        layout.addWidget(info_label)

        stats_text = (
            f"  |H_fft|² max  = {sq_max:.6e}\n"
            f"  |H_fft|² min  = {sq_min:.6e}\n"
            f"  Total frequencies: {self._n_total}"
        )
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)

        explanation = QLabel(
            "Slide to set lambda_rr (log scale).\n"
            "Frequencies with |H_fft|² >> lambda_rr are deconvolved,\n"
            "frequencies with |H_fft|² << lambda_rr are suppressed."
        )
        layout.addWidget(explanation)

        slider_layout = QHBoxLayout()
        left_label = QLabel(f"1e{self._log_min}")
        slider_layout.addWidget(left_label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(_SLIDER_STEPS)
        self._slider.setValue(_SLIDER_STEPS // 2)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(_SLIDER_STEPS // 10)
        slider_layout.addWidget(self._slider)

        right_label = QLabel(f"1e{self._log_max}")
        slider_layout.addWidget(right_label)

        layout.addLayout(slider_layout)

        self._result_label = QLabel()
        layout.addWidget(self._result_label)

        self._detail_label = QLabel()
        layout.addWidget(self._detail_label)

        self._slider.valueChanged.connect(self._update_preview)
        self._update_preview()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _slider_to_lambda(self):
        t = self._slider.value() / _SLIDER_STEPS
        log_val = self._log_min + t * (self._log_max - self._log_min)
        return 10 ** log_val

    def _update_preview(self):
        lambda_rr = self._slider_to_lambda()

        n_kept = (self._S_sq >= lambda_rr).sum().item()
        n_suppressed = self._n_total - n_kept
        pct_kept = n_kept / self._n_total * 100

        self._result_label.setText(f"lambda_rr = {lambda_rr:.4e}")
        self._detail_label.setText(
            f"  Frequencies kept:       {n_kept} / {self._n_total}  ({pct_kept:.1f}%)\n"
            f"  Frequencies suppressed: {n_suppressed} / {self._n_total}"
        )

    def selected_lambda_rr(self):
        return self._slider_to_lambda()
