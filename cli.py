"""
Centralized CLI entry point for all inverse problems.

Usage:
    matirf gui                  # launch matirf GUI
    matirf cli -c config.toml   # launch matirf CLI
    matirf reset                # delete matirf cached config.toml
    deconv gui                  # launch deconv GUI
    deconv reset                # delete deconv cached config.toml
    list                        # list available inverse problems
    help                        # show available commands
    settings show               # display current settings
    settings set <key> <value>  # change a setting
    settings reset [<key>]      # reset one or all settings to defaults
    settings gui                # open settings GUI

Each command name (matirf, deconv, ...) is auto-detected from sys.argv[0].
A new inverse problem is discovered automatically if it has a main.py
and a cache/ directory with a config.toml.
"""

import sys
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Commands that are not inverse problem names
UTILITY_COMMANDS = {"cli", "list", "help", "settings", "inverse-problems", "__main__"}


# Auto-discover inverse problem packages: any subdirectory with a main.py
def _discover_problems():
    problems = {}
    for d in sorted(PROJECT_ROOT.iterdir()):
        if d.is_dir() and (d / "main.py").exists() and (d / "__init__.py").exists():
            problems[d.name] = d
    return problems


def _get_command_name():
    """Extract the command name from sys.argv[0]."""
    return Path(sys.argv[0]).stem


def _print_help(problems):
    print("matirf_python_plus - Generic inverse problem solver\n")
    print("Commands:\n")
    for name in problems:
        print(f"  {name} gui             Launch the graphical interface")
        print(f"  {name} cli             Launch the command-line interface")
        print(f"  {name} reset           Delete the cached config.toml")
        print()
    print(f"  list                   List available inverse problems")
    print(f"  settings show          Display current settings")
    print(f"  settings set <k> <v>   Change a setting")
    print(f"  settings reset [<k>]   Reset one or all settings to defaults")
    print(f"  settings gui           Open settings GUI")
    print(f"  help                   Show this help message")


def _print_list(problems):
    print("Available inverse problems:\n")
    for name, path in problems.items():
        config_path = path / "cache" / "config.toml"
        status = "config cached" if config_path.exists() else "no config"
        print(f"  {name:<20s} ({status})")


def _reset_problem(problem_name, problem_path):
    config_path = problem_path / "cache" / "config.toml"
    if config_path.exists():
        config_path.unlink()
        print(f"Deleted {config_path}")
    else:
        print(f"No config to reset for '{problem_name}' (file not found: {config_path})")


def _handle_settings(args):
    from common.settings import _settings

    if not args or args[0] == "show":
        print(_settings.show())
        return

    if args[0] == "set":
        if len(args) < 3:
            print("Usage: settings set <key> <value>")
            sys.exit(1)
        key, value = args[1], args[2]
        try:
            _settings.set(key, value)
            print(f"  {key} = {_settings.get(key)}")
        except (KeyError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    if args[0] == "reset":
        key = args[1] if len(args) > 1 else None
        try:
            _settings.reset(key)
            if key:
                print(f"  {key} reset to {_settings.get(key)}")
            else:
                print("  All settings reset to defaults.")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    if args[0] == "gui":
        from common.settings.gui import open_settings_gui
        open_settings_gui()
        return

    print(f"Unknown settings command: '{args[0]}'")
    print("Usage: settings show | set <key> <value> | reset [<key>] | gui")
    sys.exit(1)


def main():
    problems = _discover_problems()
    cmd = _get_command_name()

    # Utility commands: list, help, settings
    if cmd in UTILITY_COMMANDS:
        arg = sys.argv[1] if len(sys.argv) > 1 else cmd
        if arg == "list":
            _print_list(problems)
        elif cmd == "settings" or arg == "settings":
            if cmd == "settings":
                _handle_settings(sys.argv[1:])
            else:
                _handle_settings(sys.argv[2:])
        else:
            _print_help(problems)
        return

    # Problem name specified but not found
    if cmd not in problems:
        print(f"Unknown inverse problem: '{cmd}'")
        print(f"Available: {', '.join(problems.keys())}")
        print(f"Run 'help' for usage information.")
        sys.exit(1)

    problem_path = problems[cmd]
    args = sys.argv[1:]

    # "matirf reset"
    if args and args[0] == "reset":
        _reset_problem(cmd, problem_path)
        return

    # "matirf help"
    if args and args[0] == "help":
        print(f"Usage:  {cmd} gui | cli | reset | help")
        return

    # "matirf gui" or "matirf cli ..." — delegate to the problem's main.py
    problem_main = importlib.import_module(f"{cmd}.main")
    problem_main.main()


if __name__ == "__main__":
    main()
