"""Extract verifiable facts from native EAGLE XML schematics."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def _outputs(source: Path, evidence: Path | None, render: Path | None) -> list[Path]:
    if render and render.suffix.lower() != ".png":
        raise ValueError("--render must name a PNG file; its companion SVG is saved alongside it")
    paths = ([render, render.with_suffix(".svg")] if render else []) + ([evidence] if evidence else [])
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(paths) or source.resolve() in resolved:
        raise ValueError("Output paths must be distinct and must not alias the source")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("Output exists, including a companion SVG; choose new paths")
    return paths


def _publish_new(artifacts: dict[Path, Path]) -> None:
    """Publish same-filesystem staged files exclusively, undoing partial success."""
    published = []
    try:
        for destination, staged in artifacts.items():
            os.link(staged, destination)
            published.append((destination, staged))
    except OSError:
        for destination, staged in reversed(published):
            # Do not remove a file another writer replaced after publication.
            if destination.exists() and os.path.samestat(destination.lstat(), staged.stat()):
                destination.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, help="save JSON evidence; refuses existing files")
    parser.add_argument("--render", type=Path, help="render source geometry to PNG")
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        destinations = _outputs(args.source, args.out, args.render)
        from .corpus import parse_eagle
        evidence = parse_eagle(args.source)
        if not 1 <= args.page <= len(evidence["sheets"]):
            raise ValueError("Page is outside this schematic")
        encoded = json.dumps(evidence, indent=2) + "\n"
        with ExitStack() as stack:
            staging = {}
            for destination in destinations:
                parent = destination.parent.resolve()
                if parent not in staging:
                    parent.mkdir(parents=True, exist_ok=True)
                    staging[parent] = Path(stack.enter_context(TemporaryDirectory(
                        prefix=".schematic-inspect-", dir=parent)))
            artifacts = {}
            if args.render:
                from .render_eagle import render_eagle
                rendered = staging[args.render.parent.resolve()] / "drawing.png"
                render_eagle(args.source, args.page, rendered)
                artifacts[args.render] = rendered
                artifacts[args.render.with_suffix(".svg")] = rendered.with_suffix(".svg")
            if args.out:
                saved = staging[args.out.parent.resolve()] / "evidence.json"
                saved.write_text(encoded, encoding="utf-8")
                artifacts[args.out] = saved
            _publish_new(artifacts)
        if not args.out:
            print(encoded, end="")
    except (ValueError, OSError) as exc:
        print(f"Schematic inspection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
