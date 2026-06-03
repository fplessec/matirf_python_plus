from PyQt5.QtWidgets import QVBoxLayout, QGroupBox

from gui.control_window.sections.add_noise_section.add_noise_ui_dictionary import ADD_NOISE_PARAMETERS_UI
from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator
from deconv.in_out import load_or_create_toml
from deconv.cache import update_cache


## add-noise parameter section for the deconv problem:
class DeconvAddNoiseSection(QGroupBox):

    def __init__(self, parent):
        super().__init__("Add noise to measurement")
        self.parent = parent
        self.params_ui_dict = ADD_NOISE_PARAMETERS_UI
        self.parameter_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        first = True
        for param_name, param_config in self.params_ui_dict.items():
            if not first:
                layout.addWidget(QSeparator('H'))
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"],
                toml_key_list=['add-noise', param_name],
                update_cache_fn=update_cache,
                load_toml_fn=load_or_create_toml,
            )
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first = False
        self.setLayout(layout)

    def update_ui_from_toml(self, toml_path):
        for param_name in self.params_ui_dict:
            self.parameter_widgets[param_name].update_ui_from_toml(toml_path)
