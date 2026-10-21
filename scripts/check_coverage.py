"""Run the CPU suite and enforce the configured branch-coverage floor."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="schematic-coverage-") as directory:
        coverage = [sys.executable, "-m", "coverage"]
        data_file = f"--data-file={Path(directory) / 'coverage'}"
        tests = subprocess.run(
            [*coverage, "run", data_file, "-m", "pytest", "-q"],
            cwd=root,
            check=False,
        )
        if tests.returncode:
            return tests.returncode
        return subprocess.run([*coverage, "report", data_file], cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
