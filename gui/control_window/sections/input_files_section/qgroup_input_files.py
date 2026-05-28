from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QHBoxLayout, QGroupBox, QVBoxLayout, QLabel

from .qwidget_tif_file_selector import TifFileSelector
from .qwidget_json_file_selector import JsonFileSelector
from gui.more_widgets import QSeparator, QSwitchButton
from core import DataMode
from in_out import load_or_create_toml, CONFIG_PATH
from cache import update_cache
from settings import FontSize


class InputFilesSection(QGroupBox):
    """
    This section of the user interface allows the user to select the path of the data.
    There is two possible modes:
        > mode_is_real == True: this corresponds to DataMode.REAL
                the data are real measurements from MA-TIRF microscopy ; the user needs to choose a tif file which is
                the actual MA-TIRF image stack, and a json file which is the parameters of this measurement.
        > mode_is_real == False: this corresponds to DataMode.SYNTHETIC
                the data are synthetic measurements ; the user needs to choose a tif file which is the ground truth
                object, a 3D image, and a json file which is a fake measurement parameters file, in order to compute
                the operator of a fake MA-TIRF measurement

    This QGroupBox focus on registering what type of data is the desired TIF file path:
    MA-TIRF multi-stack data or synthetic 3D truth, and registers 'real-data' or 'synthetic-data' inside the
    [input-paths][mode] key of the cached config.toml file.
    This QGroupBox contains all the widget to register the entirety of the parameter set [input-paths].
    """
    def __init__(self, parent=None):
        super().__init__("Input Files")
        # the colors depend on the QPalette:
        self.on_color = self.palette().color(QPalette.WindowText).name()
        self.off_color = self.palette().color(QPalette.PlaceholderText).name()
        self.parent = parent
        self.is_mode_real = self.get_cached_mode()
        self.setup_ui()

    @staticmethod
    def get_cached_mode():
        try:
            mode = load_or_create_toml(CONFIG_PATH)['input-paths']['mode']  # <- from cache
            return mode == DataMode.REAL.value
        except (KeyError, FileNotFoundError):
            return True  # default is true

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        # creation of the layouts one after another:
        top_layout = self.setup_top_layout()
        bot_layout = self.setup_bottom_layout()
        # build the layouts together to make the section:
        layout.addLayout(top_layout)
        layout.addLayout(bot_layout)
        self.setLayout(layout)

    def setup_top_layout(self):
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(4, 4, 4, 4)
        # creation of the widgets one after another:
        self.switch_button = QSwitchButton()
        if not self.is_mode_real: self.switch_button.switch_to_right()
        self.switch_button.toggled.connect(self.switch_mode)
        self.real_mode_label = QLabel("Work with real MA-TIRF measurement")
        self.synthetic_mode_label = QLabel("Simulate measurement with synthetic truth")
        self.update_labels()
        # build the widgets together to make the top layout:
        top_layout.addStretch()
        top_layout.addWidget(self.real_mode_label)
        top_layout.addWidget(self.switch_button)
        top_layout.addWidget(self.synthetic_mode_label)
        top_layout.addStretch()
        return top_layout

    def setup_bottom_layout(self):
        bot_layout = QHBoxLayout()
        bot_layout.setContentsMargins(1, 1, 1, 1)
        # creation of the widgets one after another:
        self.tif_selector = TifFileSelector(parent=self)
        self.json_selector = JsonFileSelector(parent=self)
        # build the widgets together to make the bottom layout:
        bot_layout.addWidget(self.tif_selector)
        bot_layout.addWidget(QSeparator('V'))
        bot_layout.addWidget(self.json_selector)
        return bot_layout

    def update_labels(self):
        if self.is_mode_real:
            self.real_mode_label.setStyleSheet(f"color: {self.on_color}; font-style: italic; "
                                               f"font-size: {FontSize.SMALL}pt;")
            self.synthetic_mode_label.setStyleSheet(f"color: {self.off_color}; font-style: italic; "
                                                    f"font-size: {FontSize.SMALL}pt;")
        else:
            self.real_mode_label.setStyleSheet(f"color: {self.off_color}; font-style: italic; "
                                               f"font-size: {FontSize.SMALL}pt;")
            self.synthetic_mode_label.setStyleSheet(f"color: {self.on_color}; font-style: italic; "
                                                    f"font-size: {FontSize.SMALL}pt;")

    def switch_mode(self):
        self.is_mode_real = not self.is_mode_real
        self.update_labels()
        # update the cache (utilise les .value de l'enum pour rester en sync):
        new_mode = DataMode.REAL.value if self.is_mode_real else DataMode.SYNTHETIC.value
        update_cache(['input-paths', 'mode'], new_mode)
        # update the file selectors:
        self.tif_selector.update_mode()
        self.json_selector.update_mode()

    def update_ui_from_toml(self, toml_path):
        config = load_or_create_toml(toml_path)  # <- from cache/ or any config
        mode = config['input-paths']['mode']
        self.is_mode_real = mode == DataMode.REAL.value
        self.update_labels()
        self.tif_selector.update_mode()
        self.json_selector.update_mode()
        if self.is_mode_real: self.switch_button.switch_to_left(no_signal=True)
        else: self.switch_button.switch_to_right(no_signal=True)
        self.tif_selector.update_selected_file(config['input-paths']['tif'])
        self.json_selector.update_selected_file(config['input-paths']['json'])

    def are_both_file_selected(self):
        return self.tif_selector.is_file_selected and self.json_selector.is_file_selected