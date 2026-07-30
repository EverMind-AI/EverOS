"""``everos integrations`` — installer CLI contract tests.

Pins the ``install`` / ``uninstall`` Typer commands against a fake bundle
directory and a throwaway ``HERMES_HOME``. No real bundle walk-up is
exercised — ``--source`` (or ``EVEROS_HERMES_PLUGIN_SOURCE``) short-circuits
the walk-up. Uses ``typer.testing.CliRunner`` only; no real HTTP, no real
plugins loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import integrations as integrations_mod


@pytest.fixture
def runner() -> CliRunner:
    # ``mix_stderr=False`` keeps stdout/stderr split for message assertions.
    return CliRunner()


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    src = tmp_path / "bundle"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "plugin.yaml").write_text("name: everos\n")
    return src


@pytest.fixture(autouse=True)
def _clear_source_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVEROS_HERMES_PLUGIN_SOURCE", raising=False)


# ── --help smoke ────────────────────────────────────────────────────────────


def test_install_help_exits_zero(runner: CliRunner):
    result = runner.invoke(integrations_mod.app, ["install", "--help"])
    assert result.exit_code == 0
    assert "hermes" in result.stdout


def test_uninstall_help_exits_zero(runner: CliRunner):
    result = runner.invoke(integrations_mod.app, ["uninstall", "--help"])
    assert result.exit_code == 0
    assert "hermes" in result.stdout


# ── install ─────────────────────────────────────────────────────────────────


def _target(hermes_home: Path) -> Path:
    return hermes_home / "plugins" / "everos"


def test_install_symlinks_bundle(runner: CliRunner, hermes_home: Path, bundle: Path):
    result = runner.invoke(
        integrations_mod.app, ["install", "hermes", "--source", str(bundle)]
    )
    assert result.exit_code == 0, result.output
    target = _target(hermes_home)
    assert target.is_symlink()
    assert target.resolve() == bundle.resolve()


def test_install_is_idempotent(runner: CliRunner, hermes_home: Path, bundle: Path):
    args = ["install", "hermes", "--source", str(bundle)]
    first = runner.invoke(integrations_mod.app, args)
    second = runner.invoke(integrations_mod.app, args)
    assert first.exit_code == 0
    assert second.exit_code == 0
    target = _target(hermes_home)
    assert target.is_symlink()
    assert target.resolve() == bundle.resolve()


def test_install_refuses_real_dir_without_force(
    runner: CliRunner, hermes_home: Path, bundle: Path
):
    target = _target(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "precious.txt").write_text("keep me")

    result = runner.invoke(
        integrations_mod.app,
        ["install", "hermes", "--source", str(bundle)],
        input="n\n",
    )
    assert result.exit_code == 1
    # Real dir is left untouched.
    assert not target.is_symlink()
    assert (target / "precious.txt").exists()


def test_install_force_replaces_real_dir(
    runner: CliRunner, hermes_home: Path, bundle: Path
):
    target = _target(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "old.txt").write_text("bye")

    result = runner.invoke(
        integrations_mod.app,
        ["install", "hermes", "--source", str(bundle), "--force"],
    )
    assert result.exit_code == 0, result.output
    assert target.is_symlink()
    assert target.resolve() == bundle.resolve()
    assert not (target / "old.txt").exists()


def test_install_missing_source_exits_nonzero(
    runner: CliRunner, hermes_home: Path, tmp_path: Path
):
    missing = tmp_path / "does-not-exist"
    result = runner.invoke(
        integrations_mod.app, ["install", "hermes", "--source", str(missing)]
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_install_walk_up_failure_exits_nonzero(
    runner: CliRunner,
    hermes_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Simulate a non-editable install: everos.__file__ lives somewhere with no
    # integrations/hermes ancestor, and no --source / env is supplied.
    import everos

    fake_file = tmp_path / "deep" / "everos" / "__init__.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("")
    monkeypatch.setattr(everos, "__file__", str(fake_file))

    result = runner.invoke(integrations_mod.app, ["install", "hermes"])
    assert result.exit_code != 0
    assert "could not locate" in result.output.lower()


def test_install_uses_env_source(
    runner: CliRunner, hermes_home: Path, bundle: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("EVEROS_HERMES_PLUGIN_SOURCE", str(bundle))
    result = runner.invoke(integrations_mod.app, ["install", "hermes"])
    assert result.exit_code == 0, result.output
    target = _target(hermes_home)
    assert target.is_symlink()
    assert target.resolve() == bundle.resolve()


# ── uninstall ───────────────────────────────────────────────────────────────


def test_uninstall_removes_symlink(runner: CliRunner, hermes_home: Path, bundle: Path):
    runner.invoke(integrations_mod.app, ["install", "hermes", "--source", str(bundle)])
    target = _target(hermes_home)
    assert target.is_symlink()

    result = runner.invoke(
        integrations_mod.app, ["uninstall", "hermes", "--source", str(bundle)]
    )
    assert result.exit_code == 0, result.output
    assert not target.exists()


def test_uninstall_refuses_real_dir(runner: CliRunner, hermes_home: Path, bundle: Path):
    target = _target(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()

    result = runner.invoke(
        integrations_mod.app, ["uninstall", "hermes", "--source", str(bundle)]
    )
    assert result.exit_code == 1
    assert target.is_dir()
    assert not target.is_symlink()


def test_uninstall_force_unlinks_without_ownership_check(
    runner: CliRunner, hermes_home: Path, tmp_path: Path
):
    # Symlink pointing somewhere other than the bundle.
    other = tmp_path / "other"
    other.mkdir()
    target = _target(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(other, target_is_directory=True)

    result = runner.invoke(
        integrations_mod.app,
        ["uninstall", "hermes", "--force", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert not target.exists()


def test_uninstall_refuses_mismatched_target(
    runner: CliRunner, hermes_home: Path, bundle: Path, tmp_path: Path
):
    other = tmp_path / "other"
    other.mkdir()
    target = _target(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(other, target_is_directory=True)

    result = runner.invoke(
        integrations_mod.app, ["uninstall", "hermes", "--source", str(bundle)]
    )
    assert result.exit_code == 1
    assert target.is_symlink()


def test_uninstall_nothing_to_remove(
    runner: CliRunner, hermes_home: Path, bundle: Path
):
    result = runner.invoke(
        integrations_mod.app, ["uninstall", "hermes", "--source", str(bundle)]
    )
    assert result.exit_code == 0
    assert "nothing to remove" in result.output.lower()


def test_uninstall_walk_up_failure_message(
    runner: CliRunner,
    hermes_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Symlink that dangles-resolves to a path the bundle check cannot match,
    # and walk-up cannot find a bundle either.
    import everos

    fake_file = tmp_path / "deep" / "everos" / "__init__.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("")
    monkeypatch.setattr(everos, "__file__", str(fake_file))

    other = tmp_path / "other"
    other.mkdir()
    target = _target(hermes_home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(other, target_is_directory=True)

    result = runner.invoke(integrations_mod.app, ["uninstall", "hermes"])
    assert result.exit_code == 1
    assert "could not re-resolve" in result.output.lower()
