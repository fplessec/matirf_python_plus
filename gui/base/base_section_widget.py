from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from gui.base.single_parameter_widget import SimpleParameterWidget
from gui.widgets import QSeparator


class BaseSectionWidget(QWidget):
    """
    Renders SimpleParameterWidgets from a UI dictionary backed by a TOML config.

    Can be wrapped in a QGroupBox, used inside a QStackedWidget, or subclassed.

    >> formula : callable, optional
        (values, config) -> latex. When given, a formula line is drawn — first, above the
        parameters, unless formula_position="bottom" — and redrawn whenever one changes: `values` are this section's current
        values, `config` the whole cached config (so a formula may depend on another
        section — the noise model's oracle values come from '[add-noise]'). Used to show the
        MODEL a section describes, not just its settings.

    >> changed : signal
        Emitted whenever one of the section's parameters changes, so a window can refresh
        the formulas of the sections that depend on this one.
    """

    changed = pyqtSignal()

    def __init__(self, params_ui_dict, toml_section_key, update_cache_fn, load_toml_fn,
                 config_path=None, formula=None, formula_position="top"):
        super().__init__()
        self.params_ui_dict = params_ui_dict
        self.toml_section_key = toml_section_key
        self._update_cache = update_cache_fn
        self._load_toml = load_toml_fn
        self._config_path = config_path
        self._formula = formula
        self._formula_position = formula_position
        self.formula_label = None
        self.parameter_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(1)
        self._separators = {}
        first = True
        if self._formula is not None and self._formula_position == "top":
            from gui.widgets import QLatexLabel
            self.formula_label = QLatexLabel("")
            layout.addWidget(self.formula_label, alignment=Qt.AlignCenter)
            first = False
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
        if self._formula is not None and self._formula_position == "bottom":
            from gui.widgets import QLatexLabel
            if not first:
                layout.addWidget(QSeparator('H'))
            self.formula_label = QLatexLabel("")
            layout.addWidget(self.formula_label, alignment=Qt.AlignCenter)
        layout.addStretch()
        self._setup_dependencies()
        for widget in self.parameter_widgets.values():
            self._connect_change(widget, self._on_parameter_changed)
        self.refresh_formula()

    # ── parameter dependencies (show/hide based on parent value) ────────

    ## wires depends_on: show/hide dependent widgets when a parent value changes.
    ## Each condition can be:
    ##   - a scalar / list  -> satisfied when the parent value is one of them
    ##                         (used with 'option' parents, e.g. reg -> {"reg": [...]})
    ##   - a callable(value) -> satisfied when it returns True (works with any parent type,
    ##                         e.g. a numeric 'count' -> {"count": lambda v: v > 0})
    ## With several parents, ALL conditions must hold: {"fidelity": [...], "parameters":
    ## "manual"} shows a field only for those fidelities AND manual parameters.
    def _setup_dependencies(self):
        parents = {parent for config in self.params_ui_dict.values()
                   for parent in (config.get("depends_on") or {})
                   if parent in self.parameter_widgets}
        for parent_name in parents:
            self._connect_change(self.parameter_widgets[parent_name],
                                 self._refresh_all_dependencies)
        self._refresh_all_dependencies()

    @staticmethod
    def _connect_change(widget, callback):
        slot = lambda *_: callback()
        if widget.combo is not None:
            widget.combo.currentTextChanged.connect(slot)
        elif widget.input_widget is not None:
            widget.input_widget.textChanged.connect(slot)
        elif widget.checkbox is not None:
            widget.checkbox.stateChanged.connect(slot)

    @staticmethod
    def _condition_holds(value, condition) -> bool:
        if callable(condition):
            try:
                return bool(condition(value))
            except Exception:
                return False
        required = condition if isinstance(condition, (list, tuple)) else [condition]
        return value in required

    def _is_visible(self, param_name) -> bool:
        depends_on = self.params_ui_dict[param_name].get("depends_on") or {}
        return all(self._condition_holds(self.parameter_widgets[parent].param_value, condition)
                   for parent, condition in depends_on.items()
                   if parent in self.parameter_widgets)

    ## re-evaluates all depends_on visibility (after a change, a toml load or a reset)
    def _refresh_all_dependencies(self):
        for param_name, param_config in self.params_ui_dict.items():
            if not param_config.get("depends_on"):
                continue
            visible = self._is_visible(param_name)
            self.parameter_widgets[param_name].setVisible(visible)
            sep = self._separators.get(param_name)
            if sep is not None:
                sep.setVisible(visible)

    # ── the formula line ──────────────────────────────────────────────────

    def _on_parameter_changed(self):
        self.refresh_formula()
        self.changed.emit()

    def refresh_formula(self):
        """Redraw the formula line from the current values (no-op without a formula)."""
        if self.formula_label is None:
            return
        config = {}
        if self._config_path is not None:
            try:
                config = self._load_toml(self._config_path)
            except Exception:
                config = {}
        try:
            latex = self._formula(self.get_parameters(), config)
        except Exception as error:                 # a formula must never break the window
            latex = rf"\text{{(formula unavailable: {type(error).__name__})}}"
        self.formula_label.update_latex(latex)

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
        self.refresh_formula()

    def update_ui_from_toml(self, toml_path):
        for param_name in self.params_ui_dict:
            self.parameter_widgets[param_name].update_ui_from_toml(toml_path)
        self._refresh_all_dependencies()
        self.refresh_formula()
