from PyQt5.QtWidgets import QHBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QCheckBox
from PyQt5.QtGui import QFont

from settings import FontSize
from cache import update_cache
from in_out import load_or_create_toml
from gui.more_widgets import QLatexLabel


class SimpleParameterWidget(QWidget):
    """
    One-line widget made to render and choose some parameter's value.
    Its U.I. looks like that:
    [title] [some QWidget to interact] [current value display]
    Attributes:
        title (str): a sentence that describes the parameter.
        type (str): the parameter can be 3 different types ('value', 'bool', 'option').
        param_info (dict): specific to the precedent 'type' attribute and the parameter.
        fontsize (int): the size of the font used in the QLabel/QLatexLabel.
        toml_key_list (list or None): if not None, used to specify the key pathway of the parameter in the config file.
    The different types:
        > type = 'value': the parameter is a number,
        > type = 'bool': the parameter is a choice (Yes or No, True or False),
        > type = 'option': the parameter is a choice between a finite number of options (Choice1 or Choice2 or Choice3
          ...).
    Depending on the type, the param_info dictionary contains:
        > for type = 'value', param_info should be = {
            'dtype': the data type (int or float),
            'unit': the unit of the parameter value ('' for no unit, for example: 'nm' or '°'),
            'latex_name': the display widget is a QLatexLabel, it looks like this:
                          [latex_name] = [value] [unit]
                          latex_name a latex formulation of the parameter (for example: '\\text{variable name}' or
                          '\mu')
            'default': the default value of the parameter (None for no default value)
            }
        > for type = 'bool', param_info should be = {
            'default': the default value of the parameter (None is by default False)
            }
        > for type = 'option', param_info should be = {
            'options_list': a string list, which are the possible values (for example: ["string1", "string2",
                            "string3"]) the default value is the first element of the list
            }
    """
    def __init__(self, title, type, param_info={}, fontsize=FontSize.NORMAL, toml_key_list=None):
        super().__init__()
        self.title = title
        self.type = type
        self.param_info = param_info
        self.fontsize = fontsize
        self.toml_key_list = toml_key_list
        self.input_widget = None
        self.value_display = None
        self.checkbox = None
        self.combo = None
        self.value_label = None
        self.update = None  # <- to store the callback
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
                    update_cache(self.toml_key_list, self.param_value)
            except ValueError:
                # if the user doesn't respect the data-type 'dtype':
                if self.input_widget.text() != '':
                    self.value_display.update_latex(
                        f"{param_info['latex_name']} \\text{{ must be a {param_info['dtype'].__name__}}}")
                    self.param_value = None
        self.update = update_value
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
                update_cache(self.toml_key_list, self.param_value)
        self.update = update_bool
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
                update_cache(self.toml_key_list, self.param_value)
        self.update = update_option
        self.combo.currentTextChanged.connect(update_option)
        self.layout.addWidget(self.combo)

    def update_ui_from_toml(self, toml_path):
        """Updates the current U.I. to match its value from a config file."""
        if self.toml_key_list is None:
            return
        config = load_or_create_toml(toml_path)
        toml_key_list = self.toml_key_list.copy()
        while toml_key_list != []:
            try: config = config[toml_key_list.pop(0)]
            except: config = None
        new_param_value = config if config != "null" else None
        if new_param_value == {}: return
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
                update_cache(self.toml_key_list, "None")
                self.update()
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
                update_cache(self.toml_key_list, bool(self.param_info['default']))
                self.update()
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