from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QComboBox, QStackedWidget, QHBoxLayout, QPushButton

from cache import update_cache
from in_out import load_or_create_toml, CONFIG_PATH, save_toml
from algorithms import ALGORITHMS
from .qwidget_algorithm_parameters import AlgoParamsWidget, NoneAlgoWidget


class AlgorithmAndAlgoParamsSection(QGroupBox):
    """
    This section of the UI allows the user to select with a QComboBox one reconstruction algorithm among the available
    algorithms. A QStackedWidget switch between AlgoParamsWidget(s), that allows the user to select the specific
    selected algorithm's parameters. A button can be used to reset the algorithm's parameters to its default values.
    """
    def __init__(self, parent):
        super().__init__("Select Algorithm")
        self.parent = parent
        self.selected_algo_name = 'None'
        self.setup_ui()
        self.algo_combo.setCurrentText(self.selected_algo_name)
        self.stacked_widget.setCurrentWidget(self.algo_params_ui_dict[self.selected_algo_name])

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 10, 1, 1)
        layout.setSpacing(0)
        # creation of the widgets one after another:
        first_line = self.create_first_line()
        # a QStackedWidget is used to display the parameters and changes when the selected algo is changed:
        self.stacked_widget = QStackedWidget()
        none_widget = NoneAlgoWidget()  # <- when no algo is selected
        self.stacked_widget.addWidget(none_widget)
        # creation of a widget for each algorithm, to select its parameters
        self.algo_params_ui_dict = {algo_name: AlgoParamsWidget(algo_dict['ui_params'])
                                    for algo_name, algo_dict in ALGORITHMS.items()}
        self.algo_params_ui_dict['None'] = none_widget
        for name, algo_params_ui in self.algo_params_ui_dict.items():
            self.stacked_widget.addWidget(algo_params_ui)
        # build the widgets together to make the section:
        layout.addLayout(first_line)
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

    def create_first_line(self):
        first_line = QHBoxLayout()
        first_line.setContentsMargins(0, 0, 0, 0)
        # a QComboBox to choose which algorithm i want to select:
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['None'] + list(ALGORITHMS.keys()))
        self.algo_combo.currentIndexChanged.connect(self.switch_algo)
        # a button for resetting default parameter values:
        self.reset_params_button = QPushButton("reset parameters")
        self.reset_params_button.clicked.connect(self.reset_params_to_default)
        self.update_reset_button()
        # build the widgets together to make the first line:
        first_line.addWidget(self.algo_combo, stretch=3)
        first_line.addStretch(stretch=1)
        first_line.addWidget(self.reset_params_button, stretch=1)
        return first_line
    def update_reset_button(self):
        self.reset_params_button.setDisabled(self.selected_algo_name == 'None')
    def reset_params_to_default(self):
        selected_algo_widget = self.algo_params_ui_dict[self.selected_algo_name]
        selected_algo_widget.reset_default_values()

    def switch_algo(self):
        self.selected_algo_name = self.algo_combo.currentText()
        self.stacked_widget.setCurrentWidget(self.algo_params_ui_dict[self.algo_combo.currentText()])
        self.update_reset_button()
        # writes in the cached config file the new selected algorithm
        update_cache(['algorithm'], self.selected_algo_name)
        selected_algo_widget = self.algo_params_ui_dict[self.selected_algo_name]
        # replaces all the previous algo-parameters with the new selected algorithm's algo-parameters
        config = load_or_create_toml(CONFIG_PATH)
        config['algo-params'] = selected_algo_widget.get_parameters()
        save_toml(config, CONFIG_PATH)
        # update the ui when algo switch:
        self.parent.load_cached_config()

    def update_ui_from_toml(self, toml_path):
        selected_algo_name = load_or_create_toml(toml_path)['algorithm']
        self.selected_algo_name = selected_algo_name
        selected_algo_widget = self.algo_params_ui_dict[selected_algo_name]
        selected_algo_widget.update_ui_from_toml(toml_path)
        self.stacked_widget.setCurrentWidget(self.algo_params_ui_dict[selected_algo_name])
        self.algo_combo.setCurrentText(selected_algo_name)
        self.update_reset_button()

