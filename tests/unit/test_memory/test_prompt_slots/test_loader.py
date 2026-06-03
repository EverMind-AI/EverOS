"""``PromptLoader.load`` — returns template iff enabled + non-empty."""

from __future__ import annotations

from pathlib import Path

from everos.memory.prompt_slots.loader import PromptLoader


def _write_slot(root: Path, name: str, content: str) -> None:
    slot_dir = root / "prompt_slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    (slot_dir / f"{name}.yaml").write_text(content)


def test_returns_none_when_file_missing(tmp_path: Path) -> None:
    loader = PromptLoader(tmp_path)
    assert loader.load("boundary_detection") is None


def test_returns_none_when_disabled(tmp_path: Path) -> None:
    _write_slot(tmp_path, "x", "enabled: false\ntemplate: 'hello'\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") is None


def test_returns_none_when_enabled_key_absent(tmp_path: Path) -> None:
    _write_slot(tmp_path, "x", "template: 'hello'\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") is None


def test_returns_none_when_template_empty(tmp_path: Path) -> None:
    _write_slot(tmp_path, "x", "enabled: true\ntemplate: ''\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") is None


def test_returns_none_when_template_whitespace(tmp_path: Path) -> None:
    _write_slot(tmp_path, "x", "enabled: true\ntemplate: '   '\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") is None


def test_returns_none_when_template_missing(tmp_path: Path) -> None:
    _write_slot(tmp_path, "x", "enabled: true\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") is None


def test_returns_template_when_enabled_and_non_empty(tmp_path: Path) -> None:
    _write_slot(tmp_path, "x", "enabled: true\ntemplate: 'detect now'\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") == "detect now"


def test_template_must_be_string(tmp_path: Path) -> None:
    """Non-string ``template`` (e.g. accidental int) is treated as None."""
    _write_slot(tmp_path, "x", "enabled: true\ntemplate: 42\n")
    loader = PromptLoader(tmp_path)
    assert loader.load("x") is None
