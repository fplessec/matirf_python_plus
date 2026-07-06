from common.gui.base.base_section_qgroup import BaseSectionQGroup
from .operator_parameters_ui_dictionary import OPERATOR_PARAMETERS_UI


class OperatorParametersSection(BaseSectionQGroup):
    """
    This section of the U.I. allows the user to select the set of parameter to represent our desired reconstructed
    image 'inside' the operator, this set of parameter is named 'oper-params' in the config.toml file.

    This QGroup focus on registering the desired operator parameters set inside the [oper-params] set key of the cached
    config.toml file.
    """

    title = "Operator Parameters"
    params_ui_dict = OPERATOR_PARAMETERS_UI
    toml_section_key = 'oper-params'
