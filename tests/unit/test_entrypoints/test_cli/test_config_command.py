"""``everos config show`` prints what every LLM task resolves to.

The ``[llm.<task>]`` sections are not dumped as raw nested values: each task
shows its model, endpoint host and every field grouped by the section it came
from -- the view the server logs at startup -- with the key masked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.entrypoints.cli.main import app


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def _show(tmp_path: Path, toml: str) -> str:
    root = tmp_path / "root"
    root.mkdir()
    (root / "everos.toml").write_text(toml, encoding="utf-8")
    result = CliRunner().invoke(app, ["config", "show", "--root", str(root)])
    assert result.exit_code == 0, result.output
    return result.output


def test_each_task_shows_its_model_and_where_each_field_came_from(
    tmp_path: Path,
) -> None:
    output = _show(
        tmp_path,
        '[llm]\napi_key = "sk-test-0123456789"\n'
        '[llm.boundary]\nmodel = "openai/gpt-4.1-nano"\n'
        '[llm.profile]\nbase_url = "http://localhost:8000/v1"\n',
    )
    tasks = output[output.index("[llm tasks]") :]

    assert "boundary: model = openai/gpt-4.1-nano, host = openrouter.ai" in tasks
    assert "llm.boundary: model=openai/gpt-4.1-nano" in tasks
    assert "profile: model = openai/gpt-4.1-mini, host = localhost" in tasks
    assert "decider: model = openai/gpt-4.1-mini" in tasks
    assert "timeout_seconds=60.0" in tasks, "the decider's own default is shown"


def test_the_key_is_masked_everywhere(tmp_path: Path) -> None:
    output = _show(tmp_path, '[llm]\napi_key = "sk-test-0123456789"\n')
    assert "sk-test-0123456789" not in output


def test_the_llm_section_does_not_dump_the_task_sections(tmp_path: Path) -> None:
    output = _show(tmp_path, '[llm.boundary]\nmodel = "nano"\n')
    llm = output[output.index("[llm]") : output.index("[multimodal]")]
    assert "boundary" not in llm


def test_the_deprecated_decider_keys_are_pointed_out(tmp_path: Path) -> None:
    output = _show(tmp_path, "[decider]\ntimeout_seconds = 600\n")
    assert "decider: timeout_seconds=600.0" in output
    assert "move them to [llm.decider]" in output


def test_no_deprecation_note_without_them(tmp_path: Path) -> None:
    assert "[llm.decider]" not in _show(tmp_path, "[decider]\nmax_rounds = 4\n")
