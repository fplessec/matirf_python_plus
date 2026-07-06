from functools import partial
from common.gui.app import run_app
from common.in_out import load_or_create_toml
from matirf import DEFAULT_MATIRF_CONFIG
from matirf.gui.control_window import ControlWindow
from matirf.core import MaTirfPipeline


def main():
    run_app(
        control_window_class=ControlWindow,
        pipeline_class=MaTirfPipeline,
        load_config_fn=partial(load_or_create_toml, default_config=DEFAULT_MATIRF_CONFIG),
        name="MA-TIRF Reconstruction",
    )


if __name__ == "__main__":
    main()
