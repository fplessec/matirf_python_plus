import sys
from functools import partial
from common.gui.app import run_app
from common.in_out import load_or_create_toml
from matirf import DEFAULT_MATIRF_CONFIG
from matirf.gui.control_window import ControlWindow
from matirf.core import MaTirfPipeline


## sub-commands handled here, before the generic run_app dispatcher:
SYNTH_COMMANDS = {"synth", "synthetic", "gt", "ground-truth"}


def main():
    # "matirf synth" -> open the dedicated synthetic ground-truth generator GUI
    if len(sys.argv) > 1 and sys.argv[1] in SYNTH_COMMANDS:
        from matirf.synthetic.gui import main as synthetic_gui_main
        synthetic_gui_main()
        return

    run_app(
        control_window_class=ControlWindow,
        pipeline_class=MaTirfPipeline,
        load_config_fn=partial(load_or_create_toml, default_config=DEFAULT_MATIRF_CONFIG),
        name="MA-TIRF Reconstruction",
    )


if __name__ == "__main__":
    main()
