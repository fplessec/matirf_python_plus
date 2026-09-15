from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel
)

from common.gui.base.base_section_widget import BaseSectionWidget
from common.settings import FontSize


class AlgoParamsWidget(BaseSectionWidget):
    """
    Displays one algorithm's parameter set.

    ----------
    > Parameters:
    ----------

    >> algo_dict : dict
          UI parameter dictionary for this algorithm.
    >> update_cache_fn : callable
          Function(key_path, value) to persist a parameter change.
    >> load_toml_fn : callable
          Function(filepath) -> dict to load a TOML config.
    >> config_path : Path
          Path to the cached config file.
    """

    def __init__(self, algo_dict, update_cache_fn, load_toml_fn, config_path):
        super().__init__(
            params_ui_dict=algo_dict,
            toml_section_key='algo-params',
            update_cache_fn=update_cache_fn,
            load_toml_fn=load_toml_fn,
            config_path=config_path,
        )


class NoneAlgoWidget(QWidget):
    """Placeholder widget when no algorithm is selected."""

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


