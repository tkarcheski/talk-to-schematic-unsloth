import subprocess
import sys


def test_help_without_loading_training_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "schematic_model.cli", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "evaluate" in result.stdout
    assert "benchmark" in result.stdout
