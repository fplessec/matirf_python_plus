"""
single_parameter_widget.py -> SimpleParameterWidget

In order to create the U.I., I chose to create a widget that registers one parameter at a time and allows the U.I.to
correctly update the config.toml file. This widget can be instantiated as many times as a parameter set contains
elements, using a for loop and a specific dictionary.

To illustrate, let's take a parameter set and see how to use SimpleParameterWidget to create a U.I. to manage this
parameter set. The parameter set in this example will be called ‘example’ and will be saved in the config.toml file
with the key ‘example’. A specific dictionary will need to be written as follows:
    EXAMPLE_UI_DICT = {
        'param_1_name': {
            'title': ...,
            'type': ...,
            'param_info': {...},
        },
        'param_2_name': ... ... etc...
    }
Then, in the PyQT object that manages our ‘example’ parameter set, we simply need to use a for loop and EXAMPLE_UI_DICT
to create a UI that records each parameter of ‘example’:
    for param_name, param_config in.EXAMPLE_UI_DICT.items():
        widget = SimpleParameterWidget(
            title=param_config["title"],
            type=param_config["type"],
            param_info=param_config["param_info"],
            toml_key_list=['example', param_name]
        )
        layout.addWidget(widget)
"""

from .single_parameter_widget import SimpleParameterWidget