"""Tests for ``EntryId`` parse / format / next_for."""

from __future__ import annotations

import datetime as dt

import pytest

from everos.core.persistence import EntryId

# ── format ───────────────────────────────────────────────────────────────


def test_format_pads_seq_to_eight_digits() -> None:
    eid = EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=1)
    assert eid.format() == "umc_20260422_00000001"


def test_format_pads_seq_at_99999999() -> None:
    eid = EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=99_999_999)
    assert eid.format() == "umc_20260422_99999999"


def test_str_uses_format() -> None:
    eid = EntryId(prefix="ep", date=dt.date(2026, 1, 1), seq=42)
    assert str(eid) == "ep_20260101_00000042"


# ── parse ────────────────────────────────────────────────────────────────


def test_parse_round_trip() -> None:
    raw = "umc_20260422_00000001"
    eid = EntryId.parse(raw)
    assert eid.prefix == "umc"
    assert eid.date == dt.date(2026, 4, 22)
    assert eid.seq == 1
    assert eid.format() == raw


def test_parse_handles_seq_above_pad_width() -> None:
    """Seq above 10**8 still parses; format emits more than 8 digits."""
    eid = EntryId.parse("umc_20260422_150000000")
    assert eid.seq == 150_000_000
    assert eid.format() == "umc_20260422_150000000"


def test_parse_accepts_legacy_four_digit_seq() -> None:
    """Pre-bump 4-digit seq strings still parse — format upgrades on round-trip."""
    eid = EntryId.parse("umc_20260422_0001")
    assert eid.seq == 1
    # format() returns the new 8-digit padding.
    assert eid.format() == "umc_20260422_00000001"


def test_parse_accepts_legacy_three_digit_seq() -> None:
    """Older 3-digit seq strings still parse cleanly."""
    eid = EntryId.parse("umc_20260422_001")
    assert eid.seq == 1
    assert eid.format() == "umc_20260422_00000001"


def test_parse_rejects_too_few_segments() -> None:
    with pytest.raises(ValueError, match="invalid entry id format"):
        EntryId.parse("umc_20260422")


def test_parse_rejects_invalid_date() -> None:
    with pytest.raises(ValueError, match="invalid date"):
        EntryId.parse("umc_2026XX22_00000001")


def test_parse_rejects_non_numeric_seq() -> None:
    with pytest.raises(ValueError, match="invalid seq"):
        EntryId.parse("umc_20260422_xxxx")


def test_parse_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError, match="empty prefix"):
        EntryId.parse("_20260422_00000001")


# ── next_for ─────────────────────────────────────────────────────────────


def test_next_for_seq_is_count_plus_one() -> None:
    eid = EntryId.next_for("umc", dt.date(2026, 4, 22), current_count=2)
    assert eid.seq == 3
    assert eid.format() == "umc_20260422_00000003"


def test_next_for_starts_at_one_when_empty() -> None:
    eid = EntryId.next_for("umc", dt.date(2026, 4, 22), current_count=0)
    assert eid.seq == 1


def test_next_for_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        EntryId.next_for("umc", dt.date(2026, 4, 22), current_count=-1)
