from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

from .image_3d_viewer import Image3DViewer
from .histogram_3d_widget import Histogram3DWidget


class ImageAndHisto3DViewer(QGroupBox):
    """
    An object that combines a 3D image viewer and a histogram viewer:
        > top: Image3DViewer for navigating and inspecting the 3D image
        > middle: controls to select:
            - histogram mode ('3D', 'Z-slice', 'XY-depth-column')
            - scale type ('Logarithmic', 'Linear')
            - patch size for XY-depth-column mode
        > bottom: Histogram3DWidget synchronized with the viewer:
            - updates histogram when slice changes (Z-slice mode)
            - updates histogram when mouse moves (XY-depth-column mode)
        > acts as a high-level widget to couple visualization and statistical analysis of 3D data
    """
    def __init__(self, image, title='title', parent=None):
        super().__init__(title)
        self.parent = parent
        self.image = image
        self.setup_ui()
        self.connect_widgets()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        # top: slice viewer:
        self.viewer = Image3DViewer(self.image)
        # middle: controls line:
        controls_layout = QHBoxLayout()
        # histogram mode combobox:
        self.mode_label = QLabel("histogram mode :")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['3D', 'Z-slice', 'XY-depth-column'])
        # scale type combobox:
        self.scale_label = QLabel("scale type :")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['Logarithmic', 'Linear'])
        # patch size combobox:
        self.patch_label = QLabel("patch size :")
        self.patch_combo = QComboBox()
        patch_sizes = [1,3,5,7,9,11,13,15,17,19,21,25,31,25,41,51]
        self.patch_combo.addItems([str(s) for s in patch_sizes])
        self.patch_combo.setCurrentText("5")  # default → p=2
        # assemble to controls line:
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
        # bottom: histogram:
        self.histogram = Histogram3DWidget(self.image, bins=64, mode='3D')
        # assemble top middle and bottom together:
        layout.addWidget(self.viewer)
        layout.addLayout(controls_layout)
        layout.addWidget(self.histogram)
        self.setLayout(layout)

    ## replaces the image data and updates both the viewer and histogram in-place:
    def set_image(self, image):
        self.image = image
        self.viewer.set_image(image)
        self.histogram.set_image(image)

    def connect_widgets(self):
        # combobox:
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.scale_combo.currentTextChanged.connect(self.on_scale_changed)
        self.patch_combo.currentTextChanged.connect(self.on_patch_changed)
        # slice viewer callbacks:
        self.viewer.slice_changed_callback = self.on_slice_changed
        self.viewer.mouse_moved_callback = self.on_mouse_moved

    def on_mode_changed(self, text):
        self.histogram.set_mode(text)

    def on_scale_changed(self, text):
        if text == 'Logarithmic':
            self.histogram.set_scale('log')
        else:
            self.histogram.set_scale('linear')

    def on_patch_changed(self, text):
        size = int(text)
        p = (size - 1) // 2
        self.histogram.patch_radius = p
        self.histogram.bar_container = None  # -> reset bars for consistency
        if self.histogram.mode == 'XY-depth-column':
            self.histogram.update_histogram()

    def on_slice_changed(self, value):
        if self.histogram.mode == 'Z-slice':
            self.histogram.set_slice(value)

    def on_mouse_moved(self, x, y):
        if self.histogram.mode != 'XY-depth-column':
            return
        if x is None or y is None:
            self.histogram.current_x = None
            self.histogram.current_y = None
            self.histogram.update_histogram()
        else:
            self.histogram.set_xy(x, y)


if __name__=="__main__":  # test
    import sys
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory

    import gui.more_widgets as more_widgets
    from in_out import load_tif
    import settings


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette

    package_path = Path(more_widgets.__file__).parent
    image3d = load_tif(package_path / "_image_for_test.TIF")

    window = ImageAndHisto3DViewer(image=image3d, title='test of object: ImageAndHisto3DViewer')
    window.resize(600, 900)
    window.show()

    sys.exit(app.exec_())