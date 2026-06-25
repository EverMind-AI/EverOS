"""``everos init`` — CLI behavior + edge cases.

Covers:

- default ``~/.everos/`` root with ``everos.toml`` + ``ome.toml``
- ``--root <path>`` creates target dir and both files
- ``--force`` overwrites; without it the command exits 1
- ``--print`` writes everos.toml template to stdout, NOT to disk
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.entrypoints.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip EVEROS_* env vars and move CWD away from any config file."""
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def test_root_creates_both_toml_files(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "myroot"
    result = runner.invoke(app, ["init", "--root", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "everos.toml").is_file()
    assert (target / "ome.toml").is_file()


def test_created_everos_toml_matches_shipped_template(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "myroot"
    runner.invoke(app, ["init", "--root", str(target)])
    template = Path(__file__).resolve().parents[4] / "src/everos/config/default.toml"
    assert (target / "everos.toml").read_bytes() == template.read_bytes()


def test_created_ome_toml_matches_shipped_template(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "myroot"
    runner.invoke(app, ["init", "--root", str(target)])
    template = (
        Path(__file__).resolve().parents[4] / "src/everos/config/default_ome.toml"
    )
    assert (target / "ome.toml").read_bytes() == template.read_bytes()


def test_refuses_overwrite_without_force(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "myroot"
    target.mkdir()
    (target / "everos.toml").write_text("# user-edited\n")
    (target / "ome.toml").write_text("# user-edited\n")
    result = runner.invoke(app, ["init", "--root", str(target)])
    assert result.exit_code == 1
    # Original content must be preserved.
    assert (target / "everos.toml").read_text() == "# user-edited\n"


def test_force_overwrites(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "myroot"
    target.mkdir()
    (target / "everos.toml").write_text("# user-edited\n")
    (target / "ome.toml").write_text("# user-edited\n")
    result = runner.invoke(app, ["init", "--root", str(target), "--force"])
    assert result.exit_code == 0
    # Content is now the shipped template, not the user edit.
    assert (target / "everos.toml").read_text() != "# user-edited\n"


def test_print_writes_stdout_not_disk(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--print"])
    assert result.exit_code == 0
    # Output contains shipped default.toml content.
    assert "[sqlite]" in result.output
    assert "[api]" in result.output
    # No disk side-effect in tmp cwd.
    assert not (tmp_path / "everos.toml").exists()


def test_partial_overwrite_skips_existing(runner: CliRunner, tmp_path: Path) -> None:
    """When only one file exists, only the missing file is created."""
    target = tmp_path / "myroot"
    target.mkdir()
    (target / "everos.toml").write_text("# user-edited\n")
    result = runner.invoke(app, ["init", "--root", str(target)])
    assert result.exit_code == 0
    # everos.toml preserved, ome.toml created.
    assert (target / "everos.toml").read_text() == "# user-edited\n"
    assert (target / "ome.toml").is_file()


def test_output_shows_next_steps(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "myroot"
    result = runner.invoke(app, ["init", "--root", str(target)])
    assert "Next steps" in result.output
    assert "everos server start" in result.output


# ``os`` imported above just to keep ruff from complaining; remove if Ruff
# F401 hits.
_ = os
