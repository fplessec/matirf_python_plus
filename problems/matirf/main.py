import sys
from functools import partial
from gui.app import run_app
from fileio import load_or_create_toml
from problems.matirf import DEFAULT_MATIRF_CONFIG
from problems.matirf.gui.control_window import ControlWindow
from pipeline import pipeline_for
from problems.matirf import MATIRF


## sub-commands handled here, before the generic run_app dispatcher:
SYNTH_COMMANDS = {"synth", "synthetic", "gt", "ground-truth"}


def main():
    # "matirf synth" -> open the dedicated synthetic ground-truth generator GUI
    if len(sys.argv) > 1 and sys.argv[1] in SYNTH_COMMANDS:
        from problems.matirf.synthetic.gui import main as synthetic_gui_main
        synthetic_gui_main()
        return

    run_app(
        control_window_class=ControlWindow,
        pipeline_class=pipeline_for(MATIRF),
        load_config_fn=partial(load_or_create_toml, default_config=DEFAULT_MATIRF_CONFIG),
        name="MA-TIRF Reconstruction",
    )


if __name__ == "__main__":
    main()
