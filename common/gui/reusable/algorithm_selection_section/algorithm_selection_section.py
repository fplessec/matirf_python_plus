from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QComboBox, QStackedWidget,
)

from .algo_parameters_widget import AlgoParamsWidget, NoneAlgoWidget


class AlgorithmSelectionSection(QGroupBox):
    """
    Shared algorithm selection + parameters section.

    Parameters
    ----------
    parent : QWidget
        Parent control window (must have load_cached_config()).
    algorithms_dict : dict
        The ALGORITHMS registry for this problem.
    config_path : Path
        Path to the cached config file.
    update_cache_fn : callable
    load_toml_fn : callable
    save_toml_fn : callable
    """

    def __init__(self, parent, algorithms_dict, config_path,
                 update_cache_fn, load_toml_fn, save_toml_fn,
                 problem_features=None):
        super().__init__("Select Algorithm")
        self.parent = parent
        self._algorithms = algorithms_dict
        self._problem_features = problem_features
        self._config_path = config_path
        self._update_cache = update_cache_fn
        self._load_toml = load_toml_fn
        self._save_toml = save_toml_fn
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
            name: AlgoParamsWidget(
                algo_cls.get_ui_params(self._problem_features),
                self._update_cache,
                self._load_toml,
                self._config_path,
            )
            for name, algo_cls in self._algorithms.items()
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
        self.algo_combo.addItems(['None'] + list(self._algorithms.keys()))
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
        self._update_cache(['algorithm'], self.selected_algo_name)
        widget = self.algo_widgets[self.selected_algo_name]
        config = self._load_toml(self._config_path)
        config['algo-params'] = widget.get_parameters()
        self._save_toml(config, self._config_path)
        self.parent.load_cached_config()

    def update_ui_from_toml(self, toml_path):
        name = self._load_toml(toml_path)['algorithm']
        self.selected_algo_name = name
        widget = self.algo_widgets[name]
        widget.update_ui_from_toml(toml_path)
        self.stacked.setCurrentWidget(widget)
        self.algo_combo.setCurrentText(name)
        self._update_reset_btn()