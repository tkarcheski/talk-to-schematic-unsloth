"""Import-light entry point; each command owns its argument validation."""

import argparse
import importlib
import sys

from . import __version__

COMMANDS = {
    "evaluate": lambda: importlib.import_module("evaluate"),
    "chat": lambda: importlib.import_module("chat"),
    "train": lambda: importlib.import_module("schematic_model.training"),
    "inspect": lambda: importlib.import_module("schematic_model.inspect"),
    "corpus": lambda: importlib.import_module("schematic_model.corpus"),
    "skills": lambda: importlib.import_module("schematic_model.skill_bundle"),
    "deploy": lambda: importlib.import_module("schematic_model.deployment"),
    "benchmark": lambda: importlib.import_module("schematic_model.benchmark"),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="schematic-model", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=COMMANDS)
    parsed = parser.parse_args(args[:1])
    module = COMMANDS[parsed.command]()
    previous = sys.argv
    try:
        sys.argv = [parsed.command, *args[1:]]
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
