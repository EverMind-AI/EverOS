"""``records/_dim.py`` — the narrow reader behind every ``Vector(dim)``.

The record classes cannot call ``load_settings()``: they are imported by every
CLI subcommand, so a malformed ``everos.toml`` would take ``everos config show``
down with it. These tests pin the two properties that makes that safe — the
source order matches ``Settings``, and an unreadable source falls back instead
of raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence.records._dim import DEFAULT_DIM, configured_dim


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the reader at an empty root and strip the ambient override.

    Without this a developer machine exporting ``EVEROS_EMBEDDING__DIM`` would
    make every assertion below pass for the wrong reason.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.delenv("EVEROS_EMBEDDING__DIM", raising=False)


def test_env_wins_over_root_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = 512\n", encoding="utf-8")
    monkeypatch.setenv("EVEROS_EMBEDDING__DIM", "768")
    assert configured_dim() == 768


def test_root_toml_wins_over_packaged_default(tmp_path: Path) -> None:
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = 512\n", encoding="utf-8")
    assert configured_dim() == 512


def test_packaged_default_when_root_is_silent() -> None:
    """No env, no user config: the shipped ``default.toml`` answers."""
    assert configured_dim() == DEFAULT_DIM


def test_malformed_root_toml_falls_back(tmp_path: Path) -> None:
    """A broken file must not raise — `everos config show` has to keep working.

    The disagreement it can cause is caught at startup by
    ``infra.persistence.index``, not here.
    """
    (tmp_path / "everos.toml").write_text("this is not toml {{{", encoding="utf-8")
    assert configured_dim() == DEFAULT_DIM


def test_non_integer_falls_back(tmp_path: Path) -> None:
    (tmp_path / "everos.toml").write_text(
        '[embedding]\ndim = "wide"\n', encoding="utf-8"
    )
    assert configured_dim() == DEFAULT_DIM


def test_zero_and_negative_fall_back(tmp_path: Path) -> None:
    """``Field(ge=1)`` guards Settings; the narrow reader has to guard itself."""
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = 0\n", encoding="utf-8")
    assert configured_dim() == DEFAULT_DIM
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = -8\n", encoding="utf-8")
    assert configured_dim() == DEFAULT_DIM


def test_embedding_section_of_wrong_type_falls_back(tmp_path: Path) -> None:
    (tmp_path / "everos.toml").write_text('embedding = "nope"\n', encoding="utf-8")
    assert configured_dim() == DEFAULT_DIM


def test_does_not_build_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: no ``Settings`` construction on this path (C18).

    Sabotaging ``load_settings`` must not affect the reader. If someone later
    reaches for the full settings here, this test is what fails.
    """
    import everos.config as config

    def _explode() -> None:
        raise AssertionError("configured_dim must not build Settings")

    monkeypatch.setattr(config, "load_settings", _explode)
    assert configured_dim() == DEFAULT_DIM
