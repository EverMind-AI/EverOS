"""``everos server start`` — argument resolution + uvicorn handoff.

Uvicorn ``run`` is the external boundary and is mocked. We assert the
host/port/log_level resolution chain (CLI flag > env > default) and the
KeyboardInterrupt / OSError exit paths.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import server as server_mod
from everos.entrypoints.cli.main import app as root_app


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Mock ``uvicorn.run`` and return the kwargs it was called with."""
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(server_mod.uvicorn, "run", fake_run)
    # Strip env so default resolution path is deterministic.
    for k in ("EVEROS_HOST", "EVEROS_PORT", "EVEROS_LOG_LEVEL"):
        monkeypatch.delenv(k, raising=False)
    return captured


# Typer lifts single-command sub-apps to root; we invoke via the real
# ``everos server start`` path through the assembled root app.


def test_start_uses_default_host_port_log_level(captured: dict[str, object]) -> None:
    result = CliRunner().invoke(
        root_app, ["server", "start", "--env-file", "/nonexistent"]
    )
    assert result.exit_code == 0, result.stdout
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["log_level"] == "info"
    assert kwargs["factory"] is True
    args = captured["args"]
    assert args == ("everos.entrypoints.api.app:create_app",)


def test_start_cli_flags_override_env(
    captured: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_API__HOST", "1.2.3.4")
    monkeypatch.setenv("EVEROS_API__PORT", "9000")
    monkeypatch.setenv("EVEROS_API__LOG_LEVEL", "debug")
    result = CliRunner().invoke(
        root_app,
        [
            "server",
            "start",
            "--env-file",
            "/nonexistent",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--log-level",
            "warning",
        ],
    )
    assert result.exit_code == 0, result.stdout
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8765
    assert kwargs["log_level"] == "warning"


def test_start_falls_back_to_env_when_flags_omitted(
    captured: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_API__HOST", "10.0.0.1")
    monkeypatch.setenv("EVEROS_API__PORT", "8765")
    result = CliRunner().invoke(
        root_app, ["server", "start", "--env-file", "/nonexistent"]
    )
    assert result.exit_code == 0, result.stdout
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == "10.0.0.1"
    assert kwargs["port"] == 8765


def test_start_swallows_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server_mod.uvicorn, "run", boom)
    result = CliRunner().invoke(
        root_app, ["server", "start", "--env-file", "/nonexistent"]
    )
    # KeyboardInterrupt path returns normally — exit 0.
    assert result.exit_code == 0


def test_start_exits_one_on_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("port in use")

    monkeypatch.setattr(server_mod.uvicorn, "run", boom)
    result = CliRunner().invoke(
        root_app, ["server", "start", "--env-file", "/nonexistent"]
    )
    assert result.exit_code == 1


def test_load_env_file_missing_path_is_noop(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Function should not raise when the file does not exist.
    server_mod._load_env_file(str(tmp_path / "does-not-exist.env"))


def test_load_env_file_reads_present_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("EVEROS_TEST_DOTENV_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("EVEROS_TEST_DOTENV_VAR=loaded\n")
    server_mod._load_env_file(str(env_file))
    import os

    assert os.environ.get("EVEROS_TEST_DOTENV_VAR") == "loaded"
    monkeypatch.delenv("EVEROS_TEST_DOTENV_VAR", raising=False)
