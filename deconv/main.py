from functools import partial
from common.gui.app import run_app
from common.in_out import load_or_create_toml
from deconv import DEFAULT_DECONV_CONFIG
from deconv.gui.control_window import DeconvControlWindow
from deconv.core import DeconvPipeline


def main():
    run_app(
        control_window_class=DeconvControlWindow,
        pipeline_class=DeconvPipeline,
        load_config_fn=partial(load_or_create_toml, default_config=DEFAULT_DECONV_CONFIG),
        name="2D Deconvolution",
    )


if __name__ == "__main__":
    main()
