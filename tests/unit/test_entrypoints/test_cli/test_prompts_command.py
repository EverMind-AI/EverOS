"""``everos prompts`` prints the prompt manifest as JSON, and nothing else."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from everos.entrypoints.cli.main import app
from everos.service import prompt_manifest


def test_prompts_prints_the_manifest() -> None:
    result = CliRunner().invoke(app, ["prompts"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == prompt_manifest()
    assert "chat.episode" in result.stdout
