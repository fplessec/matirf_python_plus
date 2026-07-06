from PyQt5.QtWidgets import QVBoxLayout, QGroupBox

from .base_section_widget import BaseSectionWidget


class BaseSectionQGroup(QGroupBox):
    """
    Third level in the parameter UI hierarchy (see base_section_widget.py):

        SimpleParameterWidget   1 parameter dict   →  1 line
        BaseSectionWidget       1 ui dict           →  k lines
        BaseSectionQGroup       1 ui dict           →  k lines in a titled QGroupBox frame

    BaseSectionQGroup wraps a BaseSectionWidget inside a QGroupBox so the
    section is visually grouped with a border and a title.

    Can be used in two ways:

    1. Subclass – set title, params_ui_dict, toml_section_key as class
       attributes (e.g. AddNoiseSection, OperatorParametersSection).

    2. Direct instantiation – pass them as keyword arguments.
       This is what BaseControlWindow does when it builds sections from
       inline (title, ui_dict, toml_key) tuples.
    """

    title = ""
    params_ui_dict = {}
    toml_section_key = ""

    def __init__(self, parent, update_cache_fn, load_toml_fn, *,
                 title=None, params_ui_dict=None, toml_section_key=None,
                 config_path=None):
        _title = title if title is not None else type(self).title
        _ui_dict = params_ui_dict if params_ui_dict is not None else type(self).params_ui_dict
        _toml_key = toml_section_key if toml_section_key is not None else type(self).toml_section_key
        super().__init__(_title)
        self.parent = parent
        self.section_widget = BaseSectionWidget(
            params_ui_dict=_ui_dict,
            toml_section_key=_toml_key,
            update_cache_fn=update_cache_fn,
            load_toml_fn=load_toml_fn,
            config_path=config_path,
        )
        layout = QVBoxLayout()
        layout.addWidget(self.section_widget)
        self.setLayout(layout)

    @property
    def parameter_widgets(self):
        return self.section_widget.parameter_widgets

    def get_parameters(self):
        return self.section_widget.get_parameters()

    def reset_default_values(self):
        self.section_widget.reset_default_values()

    def update_ui_from_toml(self, toml_path):
        self.section_widget.update_ui_from_toml(toml_path)
