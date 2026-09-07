from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton
from PyQt5.QtGui import QFont

from common.settings import FontSize
from common.gui.widgets import QLatexLabel


class SimpleParameterWidget(QWidget):
    """
    One-line widget made to render and choose some parameter's value.

    --------
    > Layout :
    --------

    Without extra_button:
        [title] [input widget] [current value display]

    With extra_button:
        [title] [input widget] [current value display] [button]

    ----------
    > Parameters :
    ----------

    >> title : str
        A sentence that describes the parameter.

    >> type : str
        One of 'value', 'bool', or 'option'.
            - 'value': the parameter is a number (renders a QLineEdit + QLatexLabel).
            - 'bool':  the parameter is True/False (renders a QCheckBox).
            - 'option': the parameter is chosen from a list (renders a QComboBox).

    >> param_info : dict
        Type-specific configuration (see below).

    >> fontsize : int
        Font size for labels, default FontSize.NORMAL.

    >> toml_key_list : list or None
        Key path into the TOML config, e.g. ['algo-params', 'delta'].
        Used by update_cache_fn to know where to write the value.

    >> update_cache_fn : callable or None
        Function(key_path, value) that writes a single parameter value into
        the cached config.toml.  Called whenever the user changes the value.

    >> load_toml_fn : callable or None
        Function(filepath) -> dict that loads a TOML config file.
        Used by update_ui_from_toml to refresh the widget from a config.

    >> config_path : Path or None
        Path to the cached config.toml file.  Passed through to extra_button
        callbacks so they can read/write the config.

    >> extra_button : dict or None
        Optional dict that adds a QPushButton at the end of the parameter row.
        Defined directly in the UI dictionary of the parameter, e.g.:
            "delta": {
                "title": "Anisotropy ratio coefficient",
                "type": "value",
                "param_info": { ... },
                "extra_button": {
                    "label": "Estimate",
                    "tooltip": "Estimate delta from ...",
                    "callback": estimate_delta,
                }
            }
        Keys:
            'label'    (str)      - text displayed on the button.
            'tooltip'  (str)      - tooltip text (optional).
            'callback' (callable) - function called when the button is clicked.
                                    Signature: callback(widget, update_cache_fn,
                                                        load_toml_fn, config_path)  # <- needs to respect this signature
                                    where 'widget' is this SimpleParameterWidget.
                                    The callback receives the config context so it
                                    can read/write the config without needing
                                    problem-specific imports.

    --------------------------
    > param_info format per type :
    --------------------------

    type = 'value':
        param_info = {
            'dtype': int or float,
            'unit': str (e.g. 'nm', '' for no unit),
            'latex_name': str (LaTeX formula, e.g. '\\delta', '\\text{max\\_iter}'),
            'default': number or None,
        }

    type = 'bool':
        param_info = {
            'default': bool (True or False),
        }

    type = 'option':
        param_info = {
            'options_list': list of str (the first element is the default),
        }
    """

    def __init__(self, title, type, param_info=None, fontsize=FontSize.NORMAL, toml_key_list=None,
                 update_cache_fn=None, load_toml_fn=None, config_path=None, extra_button=None):
        super().__init__()
        if param_info is None:
            param_info = {}
        self.title = title
        self.type = type
        self.param_info = param_info
        self.fontsize = fontsize
        self.toml_key_list = toml_key_list
        self._update_cache = update_cache_fn
        self._load_toml = load_toml_fn
        self._config_path = config_path
        self._extra_button = extra_button
        self.input_widget = None
        self.value_display = None
        self.checkbox = None
        self.combo = None
        self.value_label = None
        self.update_callback = None  # <- to store the callback
        self.setup_widget()

    def setup_widget(self):
        self.layout = QHBoxLayout(self)
        self.font = QFont()
        self.font.setPointSize(self.fontsize)
        title_label = QLabel(self.title)
        title_label.setFont(self.font)
        self.layout.addWidget(title_label)
        if self.type == 'value':
            self.param_value = self.param_info['default']
            self.setup_value_widget()
        elif self.type == 'bool':
            self.param_value = bool(self.param_info['default'])
            self.setup_bool_widget()
        elif self.type == 'option':
            self.param_value = self.param_info['options_list'][0]
            self.setup_option_widget()
        # Optional extra button defined in the UI dict (e.g. "Estimate" for delta)
        # The callback receives (widget, update_cache_fn, load_toml_fn, config_path)
        # so it can read/write the config without problem-specific imports:
        if self._extra_button:
            btn = QPushButton(self._extra_button['label'])
            if 'tooltip' in self._extra_button:
                btn.setToolTip(self._extra_button['tooltip'])
            callback = self._extra_button['callback']
            btn.clicked.connect(
                lambda: callback(self, self._update_cache, self._load_toml, self._config_path))
            self.layout.addWidget(btn)

    def setup_value_widget(self):
        param_info = self.param_info
        # a QLineEdit is used for the interaction:
        self.input_widget = QLineEdit(f"{'' if self.param_value is None else self.param_value}")
        self.input_widget.setFont(self.font)
        self.layout.addWidget(self.input_widget)
        # a QLatexLabel is used to display the current value:
        self.value_display = QLatexLabel('', fontsize=self.fontsize, dpi=65)
        # a callback that update the display of the current parameter value and
        # keeps track of the value in the memory of the object:
        def update_value():
            try:
                value = param_info['dtype'](self.input_widget.text())  # <- forced casting data-type
                unit = param_info['unit']
                display_text = f"{param_info['latex_name']} = {value} \;\\text{{{unit}}}"
                self.value_display.update_latex(display_text)
                self.param_value = value
                if self.toml_key_list is not None:
                    # adding a specific action to the callback to update the parameter value in cache file:
                    self._update_cache(self.toml_key_list, self.param_value)
            except ValueError:
                # if the user doesn't respect the data-type 'dtype':
                if self.input_widget.text() != '':
                    self.value_display.update_latex(
                        f"{param_info['latex_name']} \\text{{ must be a {param_info['dtype'].__name__}}}")
                    self.param_value = None
        self.update_callback = update_value
        self.input_widget.textChanged.connect(update_value)
        # update_value_display()
        self.layout.addWidget(self.value_display)

    def setup_bool_widget(self):
        # a QCheckBox is used for the interaction:
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self.param_value)
        # a QLabel is used to display the current value:
        self.value_label = QLabel()
        self.value_label.setFont(self.font)
        # a callback that update the display of the current parameter value
        # keeps track of the value in the memory of the object:
        def update_bool():
            value = self.checkbox.isChecked()
            self.value_label.setText("true" if value else "false")
            self.param_value = value
            if self.toml_key_list is not None:
                # adding a specific action to the callback to update the parameter value in cache file:
                self._update_cache(self.toml_key_list, self.param_value)
        self.update_callback = update_bool
        self.checkbox.stateChanged.connect(update_bool)
        bool_layout = QHBoxLayout()
        bool_layout.addWidget(self.checkbox)
        bool_layout.addWidget(self.value_label)
        bool_layout.addStretch()
        self.layout.addLayout(bool_layout)

    def setup_option_widget(self):
        # a QComboBox is used for the interaction (also, the QComboBox can passively display the current value):
        self.combo = QComboBox()
        self.combo.setFont(self.font)
        self.combo.addItems(self.param_info['options_list'])
        # a callback that keeps track of the parameter value in the memory of the object:
        def update_option():
            self.param_value = self.combo.currentText()
            if self.toml_key_list is not None:
                # adding a specific action to the callback to update the parameter value in cache file:
                self._update_cache(self.toml_key_list, self.param_value)
        self.update_callback = update_option
        self.combo.currentTextChanged.connect(update_option)
        self.layout.addWidget(self.combo)

    def update_ui_from_toml(self, toml_path):
        """Updates the current U.I. to match its value from a config file."""
        if self.toml_key_list is None:
            return
        config = self._load_toml(toml_path)
        node = config
        for key in self.toml_key_list:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                # key path absent from the config -> leave the widget unchanged
                # (this is what lets a pruned '[section]' keep only some keys without
                #  the missing ones being rewritten as "None")
                return
        new_param_value = node if node != "null" else None
        if new_param_value == {}:
            return
        self.temporarily_disconnect_signals()
        try:
            if self.type == 'value':
                self.update_value_from_config(new_param_value)
            elif self.type == 'bool':
                self.update_bool_from_config(new_param_value)
            elif self.type == 'option':
                self.update_option_from_config(new_param_value)
        finally:
            self.reconnect_signals()

    def update_value_from_config(self, config_value):
        try:
            typed_value = self.param_info['dtype'](config_value)
            self.input_widget.setText(str(typed_value))
            unit = self.param_info['unit']
            display_text = f"{self.param_info['latex_name']} = {typed_value} \;\\text{{{unit}}}"
            self.value_display.update_latex(display_text)
            self.param_value = typed_value
        except (ValueError, TypeError):
            if config_value is None or config_value == "None":
                self._update_cache(self.toml_key_list, "None")
                self.update_callback()
            else:
                print(
                    f"Warning: Invalid value '{config_value}' for parameter '{self.title}'"
                    f" (expected {self.param_info['dtype'].__name__})")

    def update_bool_from_config(self, config_value):
        try:
            bool_value = bool(config_value) if config_value is not None else None
            self.checkbox.setChecked(bool_value)
            self.value_label.setText("true" if bool_value else "false")
            self.param_value = bool_value
        except (ValueError, TypeError):
            if config_value is None or config_value == "None":
                self._update_cache(self.toml_key_list, bool(self.param_info['default']))
                self.update_callback()
            else:
                print(f"Warning: Invalid boolean value '{config_value}' for parameter '{self.title}'")

    def update_option_from_config(self, config_value):
        if config_value in self.param_info['options_list']:
            index = self.param_info['options_list'].index(config_value)
            self.combo.setCurrentIndex(index)
            self.param_value = config_value
        else:
            print(
                f"Warning: Invalid option '{config_value}' for parameter '{self.title}'. "
                f"Available options: {self.param_info['options_list']}")

    def temporarily_disconnect_signals(self):
        if self.type == 'value' and self.input_widget:
            self.input_widget.blockSignals(True)
        elif self.type == 'bool' and self.checkbox:
            self.checkbox.blockSignals(True)
        elif self.type == 'option' and self.combo:
            self.combo.blockSignals(True)

    def reconnect_signals(self):
        if self.type == 'value' and self.input_widget:
            self.input_widget.blockSignals(False)
        elif self.type == 'bool' and self.checkbox:
            self.checkbox.blockSignals(False)
        elif self.type == 'option' and self.combo:
            self.combo.blockSignals(False)

