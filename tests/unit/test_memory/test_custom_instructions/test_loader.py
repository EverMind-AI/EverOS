"""``CustomInstructions.load`` returns the text iff enabled and non-blank."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.memory.custom_instructions import CustomInstructions, Operator
from tests.fakes import bundled_instructions


def _write(root: Path, operator: str, content: str) -> None:
    folder = root / "custom_instructions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{operator}.yaml").write_text(content)


def test_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert CustomInstructions(tmp_path).load("boundary_detection") is None


@pytest.mark.parametrize(
    "content",
    [
        "enabled: false\ninstructions: 'keep dates'\n",
        "instructions: 'keep dates'\n",
        "enabled: true\ninstructions: ''\n",
        "enabled: true\ninstructions: '   '\n",
        "enabled: true\n",
        "enabled: true\ninstructions: 42\n",
    ],
    ids=["disabled", "no-enabled-key", "empty", "blank", "missing", "not-a-string"],
)
def test_returns_none_unless_enabled_with_text(tmp_path: Path, content: str) -> None:
    _write(tmp_path, "reflect", content)
    assert CustomInstructions(tmp_path).load("reflect") is None


def test_returns_the_instructions_verbatim(tmp_path: Path) -> None:
    _write(tmp_path, "reflect", "enabled: true\ninstructions: 'Keep {dates}.'\n")
    assert CustomInstructions(tmp_path).load("reflect") == "Keep {dates}."


@pytest.mark.parametrize(
    "operator", ["boundary_detection", "episode_extract", "profile_extract", "reflect"]
)
def test_every_bundled_operator_ships_disabled(operator: Operator) -> None:
    assert bundled_instructions().load(operator) is None
