from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

from .image_2d_viewer import Image2DViewer
from .histogram_2d_widget import Histogram2DWidget


class ImageAndHisto2DViewer(QGroupBox):
    """
    An object that combines a 2D image viewer and a histogram viewer:
        > top: Image2DViewer for inspecting the 2D image
        > middle: controls to select:
            - histogram mode ('Full', 'XY-patch')
            - scale type ('Logarithmic', 'Linear')
            - patch size for XY-patch mode
        > bottom: Histogram2DWidget synchronized with the viewer:
            - updates histogram when mouse moves (XY-patch mode)
        > acts as a high-level widget to couple visualization and statistical analysis of 2D data
    """
    def __init__(self, image, title='title', parent=None):
        super().__init__(title)
        self.parent = parent
        self.image = image
        self._setup_ui()
        self._connect_widgets()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        ## top: image viewer:
        self.viewer = Image2DViewer(self.image)
        ## middle: controls line:
        controls_layout = QHBoxLayout()
        self.mode_label = QLabel("histogram mode :")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Full', 'XY-patch'])
        self.scale_label = QLabel("scale type :")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['Logarithmic', 'Linear'])
        self.patch_label = QLabel("patch size :")
        self.patch_combo = QComboBox()
        patch_sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 41, 51]
        self.patch_combo.addItems([str(s) for s in patch_sizes])
        self.patch_combo.setCurrentText("5")  # default -> p=2
        controls_layout.addStretch()
        controls_layout.addWidget(self.mode_label)
        controls_layout.addWidget(self.mode_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.scale_label)
        controls_layout.addWidget(self.scale_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.patch_label)
        controls_layout.addWidget(self.patch_combo)
        controls_layout.addStretch()
        ## bottom: histogram:
        self.histogram = Histogram2DWidget(self.image, bins=64, mode='Full')
        ## assemble:
        layout.addWidget(self.viewer)
        layout.addLayout(controls_layout)
        layout.addWidget(self.histogram)
        self.setLayout(layout)

    ## replaces the image data and updates both the viewer and histogram in-place:
    def set_image(self, image):
        self.image = image
        self.viewer.set_image(image)
        self.histogram.set_image(image)

    def _connect_widgets(self):
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        self.patch_combo.currentTextChanged.connect(self._on_patch_changed)
        self.viewer.mouse_moved_callback = self._on_mouse_moved

    def _on_mode_changed(self, text):
        self.histogram.set_mode(text)

    def _on_scale_changed(self, text):
        self.histogram.set_scale('log' if text == 'Logarithmic' else 'linear')

    def _on_patch_changed(self, text):
        size = int(text)
        p = (size - 1) // 2
        self.histogram.patch_radius = p
        self.histogram.bar_container = None
        if self.histogram.mode == 'XY-patch':
            self.histogram._update_histogram()

    def _on_mouse_moved(self, x, y):
        if self.histogram.mode != 'XY-patch':
            return
        if x is None or y is None:
            self.histogram.current_x = None
            self.histogram.current_y = None
            self.histogram._update_histogram()
        else:
            self.histogram.set_xy(x, y)
