from PyQt5.QtWidgets import QVBoxLayout, QGroupBox

from gui.more_widgets import QSeparator
from gui.control_window import SimpleParameterWidget
from .operator_parameters_ui_dictionary import OPERATOR_PARAMETERS_UI


class OperatorParametersSection(QGroupBox):
    """
    This section of the UI allows the user to select operator parameter values, using SimpleParameterWidget. We use a
    dedicated dictionary OPERATOR_PARAMETERS_UI that stores all the attributes for each SimpleParameterWidget.
    """
    def __init__(self, parent):
        super().__init__("Operator Parameters")
        self.parent = parent
        self.params_dict = OPERATOR_PARAMETERS_UI
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
                toml_key_list=['oper-params', param_name]
            )
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first_widget = False
        self.setLayout(layout)

    def update_ui_from_toml(self, toml_path):
        for param_name, param_config in self.params_dict.items():
            try:
                widget = self.parameter_widgets[param_name]
                widget.update_ui_from_toml(toml_path)
            except:
                print("eeee")