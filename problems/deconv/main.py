from functools import partial
from gui.app import run_app
from fileio import load_or_create_toml
from problems.deconv import DEFAULT_DECONV_CONFIG
from problems.deconv.gui.control_window import DeconvControlWindow
from pipeline import pipeline_for
from problems.deconv import DECONV


def main():
    run_app(
        control_window_class=DeconvControlWindow,
        pipeline_class=pipeline_for(DECONV),
        load_config_fn=partial(load_or_create_toml, default_config=DEFAULT_DECONV_CONFIG),
        name="2D Deconvolution",
    )


if __name__ == "__main__":
    main()
