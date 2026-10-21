import subprocess
import sys


def test_help_without_loading_training_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "schematic_model.cli", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "evaluate" in result.stdout
    assert "benchmark" in result.stdout


def test_command_dispatch_is_lazy_and_restores_arguments(monkeypatch):
    from types import SimpleNamespace
    from schematic_model import cli

    calls = []
    previous = sys.argv
    module = SimpleNamespace(main=lambda: calls.append(tuple(sys.argv)))
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: module)
    cli.main(["inspect", "--help"])
    assert calls == [("inspect", "--help")]
    assert sys.argv is previous
