from PyQt5.QtWidgets import QVBoxLayout, QGroupBox

from .add_noise_ui_dictionary import ADD_NOISE_PARAMETERS_UI
from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator


class AddNoiseSection(QGroupBox):
    """
    This section of the U.I. allows the user to select the set of parameter to choose to add some noise to the MA-TIRF
    measurement stacks ; or example in cases where the user is working with a synthetic measurement, in order to test
    the robustness of an algorithm against different noise levels.
        For example, additive Gaussian noise 'n' as follows:  g = H * f_true + n
    We use the SimpleParameterWidget class, and a dedicated dictionary ADD_NOISE_PARAMETERS_UI that stores all the
    attributes for each SimpleParameterWidget.

    This QGroup focus on registering the desired noise parameters set inside the [add-noise] set key of the cached
    config.toml file.
    """
    def __init__(self, parent):
        super().__init__("Add noise to measurement")
        self.parent = parent
        self.params_ui_dict = ADD_NOISE_PARAMETERS_UI
        self.parameter_widgets = {}  # <-to keep each widget in memory
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        first_widget = True
        for param_name, param_config in self.params_ui_dict.items():
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

    def update_ui_from_toml(self, toml_path):
        """Updates the current U.I. to match its value from a config file."""
        for param_name, param_config in self.params_ui_dict.items():
            widget = self.parameter_widgets[param_name]
            widget.update_ui_from_toml(toml_path)  # <- from cache/ or any config
