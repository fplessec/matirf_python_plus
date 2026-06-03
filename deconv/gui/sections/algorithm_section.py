from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QGroupBox, QComboBox, QStackedWidget,
)

from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator
from deconv.algorithms import ALGORITHMS
from deconv.in_out import load_or_create_toml, DECONV_CONFIG_PATH, save_toml
from deconv.cache import update_cache
from settings import FontSize


## widget displaying one algorithm's parameter set (deconv version, no delta estimation):
class DeconvAlgoParamsWidget(QWidget):

    def __init__(self, algo_dict):
        super().__init__()
        self.algo_dict = algo_dict
        self.parameter_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        first = True
        for param_name, param_config in self.algo_dict.items():
            if not first:
                layout.addWidget(QSeparator('H'))
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"],
                toml_key_list=['algo-params', param_name],
                update_cache_fn=update_cache,
                load_toml_fn=load_or_create_toml,
            )
            self.parameter_widgets[param_name] = widget
            layout.addWidget(widget)
            first = False
        layout.addStretch()

    def get_parameters(self):
        return {name: w.param_value for name, w in self.parameter_widgets.items()}

    def reset_default_values(self):
        for param_name, param_config in self.algo_dict.items():
            if param_config["type"] in ['value', 'bool']:
                default = param_config["param_info"]["default"]
            elif param_config["type"] == 'option':
                default = param_config["param_info"]["options_list"][0]
            else:
                continue
            update_cache(['algo-params', param_name], default)
            self.parameter_widgets[param_name].update_ui_from_toml(DECONV_CONFIG_PATH)

    def update_ui_from_toml(self, toml_path):
        for param_name in self.algo_dict:
            self.parameter_widgets[param_name].update_ui_from_toml(toml_path)


## placeholder widget when no algorithm is selected:
class NoneAlgoWidget(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        lbl = QLabel("No algorithm selected.")
        lbl.setStyleSheet(f"color: gray; font-style: italic; font-size: {FontSize.NORMAL}pt")
        layout.addWidget(lbl)
        layout.addStretch()
        self.setLayout(layout)

    def get_parameters(self):
        return {}

    def update_ui_from_toml(self, _):
        pass


## algorithm selection + parameters section for the deconv problem:
class DeconvAlgorithmSection(QGroupBox):

    def __init__(self, parent):
        super().__init__("Select Algorithm")
        self.parent = parent
        self.selected_algo_name = 'None'
        self._setup_ui()
        self.algo_combo.setCurrentText(self.selected_algo_name)
        self.stacked.setCurrentWidget(self.algo_widgets[self.selected_algo_name])

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 10, 1, 1)
        layout.setSpacing(0)
        layout.addLayout(self._create_first_line())
        self.stacked = QStackedWidget()
        none_widget = NoneAlgoWidget()
        self.stacked.addWidget(none_widget)
        self.algo_widgets = {
            name: DeconvAlgoParamsWidget(info['ui_params'])
            for name, info in ALGORITHMS.items()
        }
        self.algo_widgets['None'] = none_widget
        for w in self.algo_widgets.values():
            self.stacked.addWidget(w)
        layout.addWidget(self.stacked)
        self.setLayout(layout)

    def _create_first_line(self):
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['None'] + list(ALGORITHMS.keys()))
        self.algo_combo.currentIndexChanged.connect(self._switch_algo)
        self.reset_btn = QPushButton("reset parameters")
        self.reset_btn.clicked.connect(self._reset_params)
        self._update_reset_btn()
        line.addWidget(self.algo_combo, stretch=3)
        line.addStretch(stretch=1)
        line.addWidget(self.reset_btn, stretch=1)
        return line

    def _update_reset_btn(self):
        self.reset_btn.setDisabled(self.selected_algo_name == 'None')

    def _reset_params(self):
        self.algo_widgets[self.selected_algo_name].reset_default_values()

    def _switch_algo(self):
        self.selected_algo_name = self.algo_combo.currentText()
        self.stacked.setCurrentWidget(self.algo_widgets[self.selected_algo_name])
        self._update_reset_btn()
        update_cache(['algorithm'], self.selected_algo_name)
        widget = self.algo_widgets[self.selected_algo_name]
        config = load_or_create_toml(DECONV_CONFIG_PATH)
        config['algo-params'] = widget.get_parameters()
        save_toml(config, DECONV_CONFIG_PATH)
        self.parent.load_cached_config()

    def update_ui_from_toml(self, toml_path):
        name = load_or_create_toml(toml_path)['algorithm']
        self.selected_algo_name = name
        widget = self.algo_widgets[name]
        widget.update_ui_from_toml(toml_path)
        self.stacked.setCurrentWidget(widget)
        self.algo_combo.setCurrentText(name)
        self._update_reset_btn()
