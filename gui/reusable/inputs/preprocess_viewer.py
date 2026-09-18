"""
Generic "see preprocessed file" popup, shared by every inverse problem's input-paths section.

It shows the input (left) next to its preprocessed / simulated measurement (right), side by
side, for the current REAL / SYNTHETIC mode. Only problem-specific bits are passed in:

    config_path      the cached config.toml to read the mode from
    default_config   fallback config dict (robust to a freshly-reset cache)
    compute_preview  callable(config, mode) -> (left_image, right_image)
                     (e.g. <Problem>Operations.compute_preprocessing_preview)
    viewer_class     the ImageAndHisto{2,3}DViewer used to display each image
    titles           {'real': (left_title, right_title), 'synthetic': (left_title, right_title)}
    size             (width, height) of the window

It closes back to the owning FileSelector (parent.sub_window = None).
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout

from core import DataMode
from fileio import load_or_create_toml


class PreprocessViewer(QWidget):
    """
    Popup showing the input (left) and its preprocessed / simulated measurement (right),
    side by side, for the current REAL / SYNTHETIC mode.

    --------
    > Layout :
    --------
        [ left viewer (input) | right viewer (preprocessed / simulated) ]

    ----------
    > Parameters :
    ----------

    >> parent : FileSelector
        The owning selector; cleared (parent.sub_window = None) when this window closes.

    >> config_path : Path or str
        Cached config.toml read to determine the current mode ('[input-paths][mode]').

    >> default_config : dict or None
        Fallback config (robust to a freshly-reset cache).

    >> compute_preview : callable
        Function(config, mode) -> (left_image, right_image). Usually
        <Problem>Operations.compute_preprocessing_preview. May raise; the caller wraps the
        construction to report problem-specific errors.

    >> viewer_class : type
        The image viewer used for each side, e.g. ImageAndHisto2DViewer / ImageAndHisto3DViewer.

    >> titles : dict
        Left/right titles per mode:
            {
              "real":      ("g_raw = input image (raw measurement)", "g = preprocessed + noise"),
              "synthetic": ("f_true = ground truth",                 "g_synth = H * f_true (+ noise)"),
            }

    >> size : (width, height)
        Initial window size.
    """

    def __init__(self, parent, *, config_path, compute_preview, viewer_class, titles,
                 size=(800, 900), default_config=None):
        super().__init__()
        self.parent = parent
        config = load_or_create_toml(config_path, default_config)
        self.mode = DataMode.from_config(config)
        # the preview computation may raise (inconsistent files, missing params, ...);
        # the caller wraps this constructor to report problem-specific errors:
        self.left, self.right = compute_preview(config, self.mode)
        is_real = self.mode == DataMode.REAL
        self.setWindowTitle("Preview - Real measurement preprocessing" if is_real
                            else "Preview - Synthetic data simulation")
        self.resize(*size)
        left_title, right_title = titles['real' if is_real else 'synthetic']
        layout = QHBoxLayout()
        layout.addWidget(viewer_class(self.left, title=left_title))
        layout.addWidget(viewer_class(self.right, title=right_title))
        self.setLayout(layout)

    def closeEvent(self, event):
        self.parent.sub_window = None
        super().closeEvent(event)
