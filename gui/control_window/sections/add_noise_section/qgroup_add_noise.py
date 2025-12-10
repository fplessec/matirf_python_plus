from PyQt5.QtWidgets import QVBoxLayout, QGroupBox

from cache import update_cache
from gui.control_window import SimpleParameterWidget
from gui.more_widgets import QSeparator
from in_out import CONFIG_PATH
from .add_noise_ui_dictionary import ADD_NOISE_PARAMETERS_UI


class AddNoiseSection(QGroupBox):
    """
    Petit widget dans qgroup_input_files pour pouvoir avoir le choix d'ajouter du bruit à la mesure synthetique
    Par exemple un bruit additif gaussien n de la façon suivante:
    g = H * f_true + n
    """
    def __init__(self, parent):
        super().__init__("Add noise to measurement")
        self.parent = parent
        self.params_dict = ADD_NOISE_PARAMETERS_UI
        self.parameter_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        first_widget = True
        for param_name, param_config in self.params_dict.items():
            if not first_widget:
                layout.addWidget(QSeparator('H'))
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"],
                toml_key_list=['add-noise', param_name]
            )
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first_widget = False
        self.setLayout(layout)

    def update_cache(self):
        if self.parent.mode1:
            update_cache(['add-noise', 'add_noise'], False)
            pass
        else:
            for param_name, param_config in self.params_dict.items():
                widget = self.parameter_widgets[param_name]
                widget.update_ui_from_toml(CONFIG_PATH)

    def update_ui_from_toml(self, toml_path):
        for param_name, param_config in self.params_dict.items():
            widget = self.parameter_widgets[param_name]
            widget.update_ui_from_toml(toml_path)
