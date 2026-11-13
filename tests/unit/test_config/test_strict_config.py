"""Unknown configuration keys are rejected; retired ones are named and ignored.

Settings used ``extra="ignore"``: a misspelt key -- in ``everos.toml`` or an
``EVEROS_<SECTION>__<KEY>`` variable -- was dropped without a word, and the run
used a default nobody chose. Now every section rejects a key it does not have,
with the file or variable it came from and the nearest valid key.

Keys EverOS wrote itself and later removed are the deliberate exception: the
everos.toml of every store built before 2.0's clean-up, and the evaluation
launchers' environment, still carry them (a scan of the evaluation store
configs and launchers found these and no other unknown key). Each is accepted,
ignored, and logged once with what to do instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.config import Settings
from everos.config.settings import IGNORED_KEYS
from everos.core.errors import ConfigurationError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("EVEROS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))


def _root(tmp_path: Path, toml: str) -> Path:
    (tmp_path / "everos.toml").write_text(toml)
    return tmp_path


def test_the_shipped_defaults_load() -> None:
    assert Settings().decider.max_rounds == 3


def test_an_unknown_key_in_everos_toml_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path, '[llm]\nmodle = "gpt-4.1-mini"\n')

    with pytest.raises(ConfigurationError) as refused:
        Settings(_everos_root=root)

    message = str(refused.value)
    assert "[llm] modle" in message
    assert str(root / "everos.toml") in message
    assert "did you mean [llm] model" in message


def test_an_unknown_section_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path, "[serch]\nexpansion = true\n")

    with pytest.raises(ConfigurationError, match=r"\[serch\]"):
        Settings(_everos_root=root)


def test_an_unknown_key_in_a_nested_section_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path, "[memorize.async_queue]\nmax_pending = 3\n")

    with pytest.raises(ConfigurationError, match=r"\[memorize.async_queue\]"):
        Settings(_everos_root=root)


def test_an_unknown_key_from_the_environment_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_DECIDER__FIT_TO_WINDOWS", "true")

    with pytest.raises(ConfigurationError) as refused:
        Settings()

    message = str(refused.value)
    assert "EVEROS_DECIDER__FIT_TO_WINDOWS" in message
    assert "did you mean [decider] fit_to_window" in message


def test_every_unknown_key_is_reported_at_once(tmp_path: Path) -> None:
    root = _root(tmp_path, "[llm]\nmodle = 'x'\n[decider]\nmax_round = 4\n")

    with pytest.raises(ConfigurationError) as refused:
        Settings(_everos_root=root)

    message = str(refused.value)
    assert "[llm] modle" in message and "[decider] max_round" in message


def test_a_secret_is_never_echoed(tmp_path: Path) -> None:
    root = _root(tmp_path, '[llm]\napi_key = "sk-very-secret-value"\nmodle = "x"\n')

    with pytest.raises(ConfigurationError) as refused:
        Settings(_everos_root=root)

    assert "sk-very-secret-value" not in str(refused.value)


@pytest.mark.parametrize("dotted", sorted(IGNORED_KEYS))
def test_a_retired_key_is_ignored(tmp_path: Path, dotted: str) -> None:
    *sections, key = dotted.split(".")
    # A retired section (no dot) is a whole table, the way old configs carry it.
    table = f"[{'.'.join(sections)}]\n{key} = 1\n" if sections else f"[{key}]\nx = 1\n"
    root = _root(tmp_path, table)

    settings = Settings(_everos_root=root)

    assert settings.decider.max_rounds == 3  # loaded; the key had no effect


def test_the_launchers_variables_still_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """What benchmarks/.env and eval_code_pin.sh export to every server (11-11)."""
    monkeypatch.setenv("EVEROS_LLM__SEARCH_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("EVEROS_DECIDER__CODE_PIN", "EverOS-train-decfix@c9711b5")
    monkeypatch.setenv("EVEROS_DECIDER__FIT_TO_WINDOW", "true")
    monkeypatch.setenv("EVEROS_OME_MAX_CONCURRENT_RUNS", "64")  # top level: unread
    monkeypatch.setenv("EVEROS_DISABLE_OME", "1")
    monkeypatch.setenv("EVEROS_LLMMR_TRACE_DUMP", "/tmp/trace.jsonl")

    settings = Settings()

    assert settings.decider.fit_to_window is True
    assert settings.ome.enabled is False


def test_a_store_config_from_before_the_clean_up_loads(tmp_path: Path) -> None:
    """The stale keys every 11-08 / 11-11 store's everos.toml carries."""
    root = _root(
        tmp_path,
        "[llm]\nsearch_timeout_seconds = 300\n"
        "[decider]\nfull_text = true\n"
        "[knowledge.search]\nsession_lock_timeout_seconds = 360.0\n"
        "[search]\nfacts_per_episode = 3\n"
        "[observability]\nrecall_hit_threshold = 0.5\n",
    )

    Settings(_everos_root=root)


def test_free_form_sections_take_any_key(tmp_path: Path) -> None:
    root = _root(
        tmp_path,
        "[llm.extra]\nmax_tokens = 512\n"
        "[llm.extra.extra_body.thinking]\ntype = 'disabled'\n"
        "[observability.headers]\nX-Anything = 'y'\n",
    )

    settings = Settings(_everos_root=root)

    assert settings.llm.extra["max_tokens"] == 512
    assert settings.observability.headers["X-Anything"] == "y"


def test_the_cli_prints_the_reason_and_exits_3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from everos.config import load_settings
    from everos.entrypoints.cli.main import app

    monkeypatch.setenv("EVEROS_DECIDER__MAX_ROUND", "4")
    load_settings.cache_clear()

    result = CliRunner().invoke(app, ["config", "show", "--root", str(tmp_path)])

    assert result.exit_code == 3
    assert "EVEROS_DECIDER__MAX_ROUND" in result.output
    assert "did you mean [decider] max_rounds" in result.output
    load_settings.cache_clear()


@pytest.mark.parametrize(
    ("env", "toml", "named"),
    [
        ({"EVEROS_API__PORT": "abc"}, "", "[api] port (from EVEROS_API__PORT)"),
        ({}, '[memory]\ntimezone = "Mars/Olympus"\n', "[memory] timezone (from "),
        ({}, "[search]\npast_trigger_words_en = ['(']\n", "past_trigger_words_en"),
        ({}, "[llm\nmodel = 1\n", "is not valid TOML"),
        (
            {"EVEROS_LLM__API_KEY": "sk-very-secret-value"},
            "[llm]\ntimeout_seconds = -5\n",
            "[llm] timeout_seconds",
        ),
    ],
)
def test_a_bad_value_exits_3_with_the_key_and_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env: dict[str, str],
    toml: str,
    named: str,
) -> None:
    from typer.testing import CliRunner

    from everos.config import load_settings
    from everos.core.errors import InvalidConfigError
    from everos.entrypoints.cli.main import app

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if toml:
        _root(tmp_path, toml)
    load_settings.cache_clear()
    with pytest.raises(InvalidConfigError) as refused:
        load_settings()
    assert named in str(refused.value)
    assert "sk-very-secret-value" not in str(refused.value)

    load_settings.cache_clear()
    result = CliRunner().invoke(app, ["config", "show", "--root", str(tmp_path)])
    assert result.exit_code == 3
    assert named in result.output
    assert "Traceback" not in result.output
    load_settings.cache_clear()


def test_config_show_prints_every_section_with_secrets_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from everos.config import load_settings
    from everos.entrypoints.cli.main import app

    monkeypatch.setenv("EVEROS_LLM__API_KEY", "sk-very-secret-value")
    monkeypatch.setenv("EVEROS_OBSERVABILITY__LANGFUSE_SECRET_KEY", "lf-secret-value")
    load_settings.cache_clear()
    result = CliRunner().invoke(app, ["config", "show", "--root", str(tmp_path)])
    load_settings.cache_clear()

    assert result.exit_code == 0
    shown = {
        line.strip("[]") for line in result.output.splitlines() if line.startswith("[")
    }
    assert set(Settings.model_fields) <= shown
    assert "memorize.async_queue" in shown
    assert "sk-very-secret-value" not in result.output
    assert "lf-secret-value" not in result.output
