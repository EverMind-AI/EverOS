"""Tests for :class:`DecisionDailyFrontmatter`.

Lives under ``test_infra`` because the schema lives under
``infra/.../mds``. Outer layers must import from the package top-level
(``everos.infra.persistence.markdown``), not ``markdown.mds``.
"""

from __future__ import annotations

import datetime as _dt

import pytest
from pydantic import ValidationError

from everos.infra.persistence.markdown import DecisionDailyFrontmatter


def _kwargs(**overrides: object) -> dict[str, object]:
    """Minimal valid kwargs for DecisionDailyFrontmatter."""
    base: dict[str, object] = {
        "id": "decision_daily_u_jason_2026-08-26",
        "user_id": "u_jason",
        "date": _dt.date(2026, 8, 26),
    }
    base.update(overrides)
    return base


def test_classvars() -> None:
    assert DecisionDailyFrontmatter.ENTRY_ID_PREFIX == "dc"
    assert DecisionDailyFrontmatter.DIR_NAME == "decisions"
    assert DecisionDailyFrontmatter.FILE_PREFIX == "decision"
    assert DecisionDailyFrontmatter.SCOPE_DIR == "users"


def test_dir_name_is_not_dot_prefixed() -> None:
    """Decisions are user-readable, unlike ``.atomic_facts`` / ``.foresights``."""
    assert not DecisionDailyFrontmatter.DIR_NAME.startswith(".")


def test_entry_id_prefix_is_token_not_full_prefix() -> None:
    """Chassis mints ``dc_<date>_<seq>`` from the token ``dc``, not ``dc_``."""
    assert DecisionDailyFrontmatter.ENTRY_ID_PREFIX == "dc"
    assert "_" not in DecisionDailyFrontmatter.ENTRY_ID_PREFIX


def test_constructs_with_id_user_id_date() -> None:
    fm = DecisionDailyFrontmatter(**_kwargs())  # type: ignore[arg-type]
    assert fm.id == "decision_daily_u_jason_2026-08-26"
    assert fm.user_id == "u_jason"
    assert fm.date == _dt.date(2026, 8, 26)
    assert fm.type == "decision_daily"
    assert fm.file_type == "decision_daily"
    assert fm.track == "user"
    assert fm.entry_count == 0


def test_deprecated_entries_defaults_empty() -> None:
    fm = DecisionDailyFrontmatter(**_kwargs())  # type: ignore[arg-type]
    assert fm.deprecated_entries == {}


def test_missing_date_raises() -> None:
    bad = _kwargs()
    del bad["date"]
    with pytest.raises(ValidationError):
        DecisionDailyFrontmatter(**bad)  # type: ignore[arg-type]


def test_deprecated_entries_round_trip() -> None:
    fm = DecisionDailyFrontmatter(
        **_kwargs(
            deprecated_entries={"dc_20260826_001": "dc_20260826_002"},
        ),  # type: ignore[arg-type]
    )
    dumped = fm.model_dump()
    assert dumped["deprecated_entries"] == {"dc_20260826_001": "dc_20260826_002"}
    round_tripped = DecisionDailyFrontmatter.model_validate(dumped)
    assert round_tripped.deprecated_entries == {
        "dc_20260826_001": "dc_20260826_002",
    }
