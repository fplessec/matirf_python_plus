from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QMessageBox,
)

from gui import SimpleParameterWidget
from gui.more_widgets import QSeparator
from cache import update_cache
from in_out import CONFIG_PATH, load_or_create_toml, load_json
from settings import FontSize
from core.operations import estimate_delta_anisotropy_from_params


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
            # special case : for parameter 'delta' (anisotropy coefficient) I add a button 'Estimate' which uses the
            # estimate_delta_anisotropy_from_params function from core/operations.py :
            if param_name == 'delta':
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(widget, stretch=1)
                estimate_btn = QPushButton("Estimate")
                estimate_btn.setToolTip(
                    "Estimate delta from measurement parameters (.json) "
                    "and operator parameters (nz, z0, zN)."
                )
                estimate_btn.clicked.connect(self._estimate_delta)
                row_layout.addWidget(estimate_btn)
                layout.addWidget(row)
            else:
                layout.addWidget(widget)

            first_widget = False
        layout.addStretch()

    def _estimate_delta(self):
        """
        Callback that estimate delta from the selected value of the user (saved in the cache) :
        the selected values that are required to compute delta are some physical parameters
        from the json file and the z0, zN, nZ parameters from the oper-params group.
        This function displays a error message if any of this values aren't selected, and
        otherwise it updates the SimpleParameterWidget for 'delta'.
        """
        _REQUIRED_MEASUREMENT_KEYS = ['n_medium', 'numerical_aperture', 'wavelength_nm']
        _REQUIRED_OPER_KEYS = ['nz', 'z0', 'zN']
        config = load_or_create_toml(CONFIG_PATH)
        missing = []
        # is the json file selected ?
        json_path = config.get('input-paths', {}).get('json', 'None')
        if json_path == 'None' or not json_path:
            missing.append("Measurement parameters file (.json) not selected.")
            measurement_params = None
        else:
            try:
                measurement_params = load_json(json_path)
            except Exception as e:
                missing.append(f"Cannot load .json file: {type(e).__name__}: {e}")
                measurement_params = None
        # does the json file contain the required parameters ?
        if measurement_params is not None:
            for key in _REQUIRED_MEASUREMENT_KEYS:
                if key not in measurement_params or measurement_params[key] in (None, 'None'):
                    missing.append(f"Missing measurement parameter '{key}' in .json.")
        # does the oper-params group contain the required parameters ?
        oper_params = config.get('oper-params', {})
        for key in _REQUIRED_OPER_KEYS:
            if key not in oper_params or oper_params[key] in (None, 'None'):
                missing.append(f"Missing operator parameter '{key}' (set it in the operator section).")
        # if there is anything missing then a error message is display:
        if missing:
            QMessageBox.warning(
                self,
                "Cannot estimate delta",
                "The following parameters are required to estimate delta:\n\n"
                + "\n".join(f"  • {m}" for m in missing)
            )
            return
        # if not, then delta can be estimated:
        try:
            delta = estimate_delta_anisotropy_from_params(measurement_params, oper_params)
            delta = round(delta, 4)  # no need to have more than 4 decimals
        except Exception as e:
            QMessageBox.warning(
                self,
                "Cannot estimate delta",
                f"An unexpected error occurred during estimation:\n\n"
                f"{type(e).__name__}: {e}"
            )
            return
        # if the computation went fine then the cache and the U.I. are updated:
        update_cache(['algo-params', 'delta'], float(delta))
        self.parameter_widgets['delta'].update_ui_from_toml(CONFIG_PATH)

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

