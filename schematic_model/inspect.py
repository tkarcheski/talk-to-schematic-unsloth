"""Extract verifiable facts from native EAGLE XML schematics."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, help="save JSON evidence; refuses existing files")
    parser.add_argument("--render", type=Path, help="render source geometry to PNG")
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        from .corpus import parse_eagle
        evidence = parse_eagle(args.source)
        if args.out and args.out.exists() or args.render and args.render.exists():
            raise ValueError("Output exists; choose a new path")
        if not 1 <= args.page <= len(evidence["sheets"]):
            raise ValueError("Page is outside this schematic")
        if args.render:
            from .render_eagle import render_eagle
            render_eagle(args.source, args.page, args.render)
        encoded = json.dumps(evidence, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
    except (ValueError, OSError) as exc:
        print(f"Schematic inspection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
