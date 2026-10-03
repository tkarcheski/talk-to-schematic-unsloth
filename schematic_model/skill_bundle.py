"""List or export the reusable skills bundled with this package."""

import argparse
from importlib.resources import files
import json
from pathlib import Path
import shutil
import tempfile


def export_skills(destination):
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError("Skill export destination exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as stage:
        staged = Path(stage) / "skills"
        staged.mkdir()
        for resource in files("schematic_model").joinpath("skills").iterdir():
            entry = resource.joinpath("SKILL.md")
            if resource.is_dir() and entry.is_file():
                target = staged / resource.name
                target.mkdir()
                (target / "SKILL.md").write_bytes(entry.read_bytes())
        if not list(staged.iterdir()):
            raise ValueError("Package contains no skill resources")
        if destination.exists():
            raise ValueError("Skill export destination appeared during export")
        shutil.move(str(staged), str(destination))
    return sorted(str(path) for path in destination.glob("*/SKILL.md"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new directory to receive all three skills")
    args = parser.parse_args(argv)
    if args.out:
        try:
            result = export_skills(args.out)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    else:
        result = sorted(resource.name for resource in files("schematic_model").joinpath("skills").iterdir()
                        if resource.is_dir() and resource.joinpath("SKILL.md").is_file())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
