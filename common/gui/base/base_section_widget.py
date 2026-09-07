from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .single_parameter_widget import SimpleParameterWidget
from common.gui.widgets import QSeparator


class BaseSectionWidget(QWidget):
    """
    Renders SimpleParameterWidgets from a UI dictionary backed by a TOML config.

    Can be wrapped in a QGroupBox, used inside a QStackedWidget, or subclassed.
    """

    def __init__(self, params_ui_dict, toml_section_key, update_cache_fn, load_toml_fn,
                 config_path=None):
        super().__init__()
        self.params_ui_dict = params_ui_dict
        self.toml_section_key = toml_section_key
        self._update_cache = update_cache_fn
        self._load_toml = load_toml_fn
        self._config_path = config_path
        self.parameter_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        self._separators = {}
        first = True
        for param_name, param_config in self.params_ui_dict.items():
            sep = None
            if not first:
                sep = QSeparator('H')
                layout.addWidget(sep)
            widget = SimpleParameterWidget(
                title=param_config["title"],
                type=param_config["type"],
                param_info=param_config["param_info"],
                toml_key_list=[self.toml_section_key, param_name],
                update_cache_fn=self._update_cache,
                load_toml_fn=self._load_toml,
                # config_path and extra_button are forwarded to SimpleParameterWidget
                # so that extra buttons (defined in the UI dict) can read/write the
                # config without needing problem-specific imports.
                config_path=self._config_path,
                extra_button=param_config.get("extra_button"),
            )
            self.parameter_widgets[param_name] = widget
            self._separators[param_name] = sep
            layout.addWidget(widget)
            first = False
        layout.addStretch()
        self._setup_dependencies()

    # ── parameter dependencies (show/hide based on parent value) ────────

    ## wires depends_on: show/hide dependent widgets when the parent value changes.
    ## The condition can be:
    ##   - a scalar / list  -> the dependent is shown when the parent value is one of them
    ##                         (used with 'option' parents, e.g. reg -> {"reg": [...]})
    ##   - a callable(value) -> shown when it returns True (works with any parent type,
    ##                         e.g. a numeric 'count' -> {"count": lambda v: v > 0})
    def _setup_dependencies(self):
        for param_name, param_config in self.params_ui_dict.items():
            depends_on = param_config.get("depends_on")
            if not depends_on:
                continue
            for parent_name, condition in depends_on.items():
                if parent_name not in self.parameter_widgets:
                    continue
                parent_widget = self.parameter_widgets[parent_name]
                # connect the parent's signal to show/hide this dependent widget
                self._connect_dependency(parent_widget, param_name, condition)
                # apply initial visibility
                self._update_dependency_visibility(parent_widget, param_name, condition)

    def _connect_dependency(self, parent_widget, dependent_name, condition):
        callback = lambda _=None: self._update_dependency_visibility(
            parent_widget, dependent_name, condition)
        if parent_widget.combo is not None:
            parent_widget.combo.currentTextChanged.connect(callback)
        elif parent_widget.input_widget is not None:
            parent_widget.input_widget.textChanged.connect(callback)
        elif parent_widget.checkbox is not None:
            parent_widget.checkbox.stateChanged.connect(callback)

    def _update_dependency_visibility(self, parent_widget, dependent_name, condition):
        value = parent_widget.param_value
        if callable(condition):
            try:
                visible = bool(condition(value))
            except Exception:
                visible = False
        else:
            required = condition if isinstance(condition, (list, tuple)) else [condition]
            visible = value in required
        self.parameter_widgets[dependent_name].setVisible(visible)
        sep = self._separators.get(dependent_name)
        if sep is not None:
            sep.setVisible(visible)

    ## re-evaluates all depends_on visibility (after loading from toml or resetting)
    def _refresh_all_dependencies(self):
        for param_name, param_config in self.params_ui_dict.items():
            depends_on = param_config.get("depends_on")
            if not depends_on:
                continue
            for parent_name, condition in depends_on.items():
                if parent_name not in self.parameter_widgets:
                    continue
                parent_widget = self.parameter_widgets[parent_name]
                self._update_dependency_visibility(parent_widget, param_name, condition)

    # ── public helpers ────────────────────────────────────────────────────

    def get_parameters(self):
        return {name: w.param_value for name, w in self.parameter_widgets.items()}

    ## resets every parameter to its default value from the UI dict
    def reset_default_values(self):
        for param_name, param_config in self.params_ui_dict.items():
            if param_config["type"] in ('value', 'bool'):
                default = param_config["param_info"]["default"]
            elif param_config["type"] == 'option':
                default = param_config["param_info"]["options_list"][0]
            else:
                continue
            self._update_cache([self.toml_section_key, param_name], default)
            if self._config_path:
                self.parameter_widgets[param_name].update_ui_from_toml(self._config_path)
        self._refresh_all_dependencies()

    def update_ui_from_toml(self, toml_path):
        for param_name in self.params_ui_dict:
            self.parameter_widgets[param_name].update_ui_from_toml(toml_path)
        self._refresh_all_dependencies()
