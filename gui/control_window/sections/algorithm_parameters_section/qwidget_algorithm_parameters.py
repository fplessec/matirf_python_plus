from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator
from cache import update_cache
from in_out import CONFIG_PATH
from settings import FontSize


class AlgoParamsWidget(QWidget):
    """
    A widget that handle a specific algorithm's list of parameters. It should take in argument algo_dict, which is a
    dictionary that describes the algorithm's parameters in a UI view. The format of the dictionary is made to create
    SimpleParameterWidget(s).
    When implementing a new algorithm, a new dictionary should be written in the same structure as those already
    written in the algorithms.ui_params_dicts module.

    This QWidget focus on registering the desired algorithm parameters set of the chosen algorithm inside the
    [algo-params] set key of the cached config.toml file.
    """
    def __init__(self, algo_dict):
        super().__init__()
        self.algo_dict = algo_dict
        self.parameter_widgets = {}
        self.setup_ui()
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        first_widget = True
        for param_name, param_config in self.algo_dict.items():
            if not first_widget:
                layout.addWidget(QSeparator('H'))
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"],
                toml_key_list=['algo-params', param_name]
            )
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first_widget = False
        layout.addStretch()

    def get_parameters(self):
        parameters = {}
        for param_name, widget in self.parameter_widgets.items():
            parameters[param_name] = widget.param_value
        return parameters

    def reset_default_values(self):
        for param_name, param_config in self.algo_dict.items():
            if param_config["type"] in ['value', 'bool']:
                default_value = param_config["param_info"]["default"]
            elif param_config["type"] == 'option':
                default_value = param_config["param_info"]["options_list"][0]
            # write the default value in cache:
            update_cache(['algo-params', param_name], default_value)  # -> to cache
            # update ui from cache:
            widget = self.parameter_widgets[param_name]
            widget.update_ui_from_toml(CONFIG_PATH)  # <- from cache

    def update_ui_from_toml(self, toml_path):
        for param_name, param_config in self.algo_dict.items():
            widget = self.parameter_widgets[param_name]
            widget.update_ui_from_toml(toml_path)  # <- from cache/ or any config


class NoneAlgoWidget(QWidget):
    """A widget that mimics the behavior of a AlgoParamsWidget but when no algorithm is selected."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        none_label = QLabel("No algorithm selected.")
        none_label.setStyleSheet(f"color: gray; font-style: italic; font-size: {FontSize.NORMAL}pt")
        layout.addWidget(none_label)
        layout.addStretch()
        self.setLayout(layout)
    def get_parameters(self):
        return {}
    def update_ui_from_toml(self, _):
        pass

