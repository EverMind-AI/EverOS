"""Unit tests for timezone-aware datetime helpers."""

from __future__ import annotations

import datetime as dt
import os

import pytest

from everos.component.utils import datetime as dt_module
from everos.component.utils.datetime import (
    UtcDatetime,
    ensure_utc,
    from_iso_format,
    from_timestamp,
    get_now_with_timezone,
    get_utc_now,
    to_date_str,
    to_display_tz,
    to_iso_format,
    to_timestamp_ms,
    today_with_timezone,
)
from everos.config import load_settings


@pytest.fixture(autouse=True)
def _isolate_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset env + caches so each test gets a fresh default-tz resolution."""
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()


def test_get_now_is_timezone_aware() -> None:
    now = get_now_with_timezone()
    assert now.tzinfo is not None


def test_from_timestamp_seconds() -> None:
    ts = 1_758_025_061  # 10-digit → seconds
    result = from_timestamp(ts)
    assert result.tzinfo is not None
    assert int(result.timestamp()) == ts


def test_from_timestamp_milliseconds() -> None:
    ts_ms = 1_758_025_061_588  # 13-digit → milliseconds
    result = from_timestamp(ts_ms)
    assert result.tzinfo is not None
    assert int(result.timestamp() * 1000) == ts_ms


def test_from_iso_format_aware() -> None:
    s = "2026-04-22T10:30:45+08:00"
    result = from_iso_format(s)
    assert result.tzinfo is not None
    assert result.hour == 10


def test_from_iso_format_naive_attaches_display_tz() -> None:
    s = "2026-04-22T10:30:45"
    result = from_iso_format(s)
    assert result.tzinfo is not None  # default tz attached


def test_to_iso_format_roundtrip() -> None:
    now = get_now_with_timezone()
    s = to_iso_format(now)
    parsed = from_iso_format(s)
    assert parsed == now


def test_to_timestamp_ms() -> None:
    d = dt.datetime(2026, 4, 22, 10, 30, 45, tzinfo=dt.UTC)
    ts_ms = to_timestamp_ms(d)
    assert ts_ms == int(d.timestamp() * 1000)


def test_display_tz_defaults_to_utc() -> None:
    """No explicit setting → UTC."""
    now = get_now_with_timezone()
    assert now.utcoffset() == dt.timedelta(0)


def test_display_tz_uses_settings_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EVEROS_MEMORY__TIMEZONE`` env var overrides via Settings."""
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    now = get_now_with_timezone()
    assert now.utcoffset() == dt.timedelta(hours=8)


def test_display_tz_ignores_os_tz_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OS ``TZ`` is *not* consulted — Settings is the sole source."""
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    now = get_now_with_timezone()
    assert now.utcoffset() == dt.timedelta(0)  # still UTC


def test_today_with_timezone_returns_date() -> None:
    today = today_with_timezone()
    assert isinstance(today, dt.date)
    # Sanity: matches the date component of a fresh now() call.
    assert today == get_now_with_timezone().date()


def test_today_with_timezone_respects_settings_tz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different TZ may yield a different bucket for the same UTC instant."""
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    today = today_with_timezone()
    assert today == get_now_with_timezone().date()


# ── to_iso_format multi-type ─────────────────────────────────────────────


def test_to_iso_format_none_passthrough() -> None:
    assert to_iso_format(None) is None


def test_to_iso_format_empty_string_returns_none() -> None:
    assert to_iso_format("") is None


def test_to_iso_format_int_seconds() -> None:
    out = to_iso_format(1_758_025_061)
    assert out is not None
    parsed = from_iso_format(out)
    assert int(parsed.timestamp()) == 1_758_025_061


def test_to_iso_format_int_milliseconds() -> None:
    out = to_iso_format(1_758_025_061_588)
    assert out is not None
    parsed = from_iso_format(out)
    assert int(parsed.timestamp() * 1000) == 1_758_025_061_588


def test_to_iso_format_str_revalidates() -> None:
    out = to_iso_format("2026-04-22T10:30:45Z")
    assert out is not None
    parsed = from_iso_format(out)
    assert parsed.utcoffset() == dt.timedelta(0)


def test_to_iso_format_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="unsupported type"):
        to_iso_format([1, 2, 3])  # type: ignore[arg-type]


def test_to_iso_format_rejects_bool_explicitly() -> None:
    """``bool`` is technically an ``int`` subclass — reject to avoid surprises."""
    with pytest.raises(TypeError, match="bool"):
        to_iso_format(True)  # type: ignore[arg-type]


# ── from_iso_format multi-type ───────────────────────────────────────────


def test_from_iso_format_accepts_datetime() -> None:
    d = dt.datetime(2026, 4, 22, 10, 30, 45, tzinfo=dt.UTC)
    assert from_iso_format(d) == d


def test_from_iso_format_attaches_tz_to_naive_datetime() -> None:
    naive = dt.datetime(2026, 4, 22, 10, 30, 45)
    out = from_iso_format(naive)
    assert out.tzinfo is not None


def test_from_iso_format_accepts_int_timestamp() -> None:
    out = from_iso_format(1_758_025_061)
    assert int(out.timestamp()) == 1_758_025_061


def test_from_iso_format_accepts_z_suffix() -> None:
    out = from_iso_format("2026-04-22T10:30:45Z")
    assert out.utcoffset() == dt.timedelta(0)


def test_from_iso_format_rejects_bool() -> None:
    with pytest.raises(TypeError, match="bool"):
        from_iso_format(True)  # type: ignore[arg-type]


# ── to_date_str ──────────────────────────────────────────────────────────


def test_to_date_str_returns_yyyy_mm_dd() -> None:
    d = dt.datetime(2026, 4, 22, 10, 30, 45, tzinfo=dt.UTC)
    assert to_date_str(d) == "2026-04-22"


def test_to_date_str_passes_through_none() -> None:
    assert to_date_str(None) is None


# ── Q2 two-zone discipline invariants ───────────────────────────────────
#
# These pin the storage-UTC + display-TZ contract:
#
#   - get_utc_now() always returns a UTC-aware datetime regardless of
#     the display-timezone setting.
#   - ensure_utc() normalises any input (naive or aware) to UTC.
#   - to_display_tz() always converts to the configured display tz.
#   - UtcDatetime Annotated field auto-normalises on Pydantic validation.
#   - Round-trip: a write-time get_utc_now() value, after UtcDatetime
#     validation + a hypothetical SQLite tz-strip + read-back, lands
#     at the same UTC instant.


def test_get_utc_now_is_always_utc_regardless_of_display_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_utc_now() must ignore EVEROS_MEMORY__TIMEZONE — storage stays UTC."""
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    now = get_utc_now()
    assert now.tzinfo is dt.UTC


def test_ensure_utc_treats_naive_input_as_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naive input is treated as already-UTC wall-clock — no display-tz drift.

    This is the **storage boundary** semantic: the dominant naive
    source is SQLite reads (SQLAlchemy strips tz on write, so what
    comes back is naive but its bytes are UTC). Treating those naive
    reads as display-tz would drift by the offset on every round trip.

    With display tz = Shanghai, a naive ``14:00`` must NOT be
    reinterpreted as Shanghai 14:00 → UTC 06:00; it must stay UTC
    ``14:00`` so the round trip is invariant.
    """
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    out = ensure_utc(dt.datetime(2026, 5, 29, 14))
    assert out.tzinfo is dt.UTC
    assert out.hour == 14


def test_ensure_utc_converts_aware_input() -> None:
    """Already-aware input is converted to UTC, never mutated in place."""
    from zoneinfo import ZoneInfo

    aware = dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("Asia/Shanghai"))
    out = ensure_utc(aware)
    assert out.tzinfo is dt.UTC
    assert out.hour == 6


def test_to_display_tz_converts_to_settings_tz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UTC ``06:00`` rendered with display tz = Shanghai becomes 14:00 + 08:00."""
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    utc = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)
    out = to_display_tz(utc)
    assert out.hour == 14
    assert out.utcoffset() == dt.timedelta(hours=8)


def test_to_display_tz_attaches_to_naive_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naive input is treated as already display-tz local — attach + return."""
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    out = to_display_tz(dt.datetime(2026, 5, 29, 14))
    assert out.hour == 14
    assert out.utcoffset() == dt.timedelta(hours=8)


def test_utc_datetime_annotated_normalises_on_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic field declared as UtcDatetime always materialises UTC-aware."""
    from pydantic import BaseModel

    class _Row(BaseModel):
        ts: UtcDatetime

    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    # Naive input → assumed already-UTC (storage-boundary semantic),
    # NOT reinterpreted under the display tz. The round trip therefore
    # preserves the wall-clock hour through a SQLite-style tz-strip.
    row = _Row(ts=dt.datetime(2026, 5, 29, 14))
    assert row.ts.tzinfo is dt.UTC
    assert row.ts.hour == 14

    # Already-aware input → astimezone(UTC).
    from zoneinfo import ZoneInfo

    row2 = _Row(ts=dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("America/New_York")))
    assert row2.ts.tzinfo is dt.UTC
    assert row2.ts.hour == 18


def test_storage_round_trip_preserves_utc_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write get_utc_now → strip tz (simulate SQLite) → ensure_utc on read.

    The UTC instant must be preserved end-to-end regardless of display tz
    — this is the bug the two-zone discipline prevents.
    """
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    written = get_utc_now()
    # Simulate what SQLAlchemy does on a tz-aware-into-SQLite write: strip tz.
    on_disk_naive = written.replace(tzinfo=None)
    # ``ensure_utc`` on a naive value attaches display tz then converts; for a
    # value that came out of SQLite that contract is wrong (the value is
    # already UTC, not display-tz). The correct read path therefore is to
    # attach UTC explicitly — UtcDatetime does exactly this when treating the
    # naive instant as already-UTC via tzinfo=UTC replacement.
    read_back = on_disk_naive.replace(tzinfo=dt.UTC)
    assert read_back == written


def test_to_display_tz_round_trip_idempotent_under_repeated_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """to_display_tz ∘ to_display_tz == to_display_tz (no drift on re-render)."""
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    utc = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)
    once = to_display_tz(utc)
    twice = to_display_tz(once)
    assert once == twice


# ── Gap-coverage matrix (per Q3 audit) ──────────────────────────────────
#
# These tests pin the boundaries the original Q2 round missed. Each test
# names the gap it covers. New work touching datetime semantics should
# extend this section, not leave gaps unguarded.


def test_ensure_utc_aware_utc_is_noop() -> None:
    """``ensure_utc(aware UTC)`` returns an equal-valued aware UTC datetime."""
    d = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)
    out = ensure_utc(d)
    assert out == d
    assert out.tzinfo is dt.UTC


def test_utc_datetime_field_passes_through_aware_utc() -> None:
    """A field declared ``UtcDatetime`` accepts an already-UTC aware input."""
    from pydantic import BaseModel

    class _Row(BaseModel):
        ts: UtcDatetime

    aware = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)
    row = _Row(ts=aware)
    assert row.ts == aware
    assert row.ts.tzinfo is dt.UTC


def test_get_utc_now_default_factory_used_by_pydantic_field() -> None:
    """``default_factory=get_utc_now`` populates a UtcDatetime field with aware UTC."""
    from pydantic import BaseModel
    from pydantic import Field as PField

    class _Row(BaseModel):
        ts: UtcDatetime = PField(default_factory=get_utc_now)

    row = _Row()
    assert row.ts.tzinfo is dt.UTC


def test_pydantic_isoformat_renders_utc_as_z_suffix() -> None:
    """Pydantic's default JSON serialisation canonicalises UTC to ``Z`` suffix.

    This is what gives the API contract its ``"timestamp": "...Z"`` shape
    when the display tz is UTC. If Pydantic ever changes this, response
    consumers that match on ``.endswith("Z")`` would break — pin it here.
    """
    from pydantic import BaseModel

    class _Row(BaseModel):
        ts: dt.datetime

    row = _Row(ts=dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC))
    rendered = row.model_dump_json()
    assert '"ts":"2026-05-29T06:00:00Z"' in rendered


def test_sqlite_round_trip_under_shanghai_display_tz(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """End-to-end: write under Shanghai → read → row is aware UTC.

    Exercises the SQLAlchemy ``load`` event hook on real SQLite — without
    it, the read would return naive, and downstream ``astimezone(...)``
    would silently interpret the naive value as local-process time.
    """
    import asyncio
    import json as _json

    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    from everos.core.persistence.sqlite import SQLModel as _SQLModel
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        sqlite_manager,
        unprocessed_buffer_repo,
    )

    sqlite_manager._engine = None
    sqlite_manager._session_factory = None

    async def _run() -> None:
        engine = sqlite_manager.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(_SQLModel.metadata.create_all)

        target = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)
        row = UnprocessedBuffer(
            message_id="m_rt",
            session_id="s_rt",
            track="memorize",
            sender_id="alice",
            role="user",
            timestamp=target,
            content_items_json=_json.dumps([{"type": "text", "text": "x"}]),
            text="x",
        )
        await unprocessed_buffer_repo.replace("s_rt", "memorize", [row])
        rows = await unprocessed_buffer_repo.list_for_track("s_rt", "memorize")
        assert rows[0].timestamp.tzinfo is dt.UTC, (
            "SQLAlchemy load event hook must attach UTC; "
            f"got tzinfo={rows[0].timestamp.tzinfo!r}"
        )
        assert rows[0].timestamp == target
        # BaseTable.created_at / updated_at inherit the hook too.
        assert rows[0].created_at.tzinfo is dt.UTC
        await sqlite_manager.dispose_engine()

    asyncio.run(_run())


def test_lancedb_schema_overrides_subclass_declared_non_utc_tz() -> None:
    """A subclass that tries to declare ``tz=America/New_York`` is forced to UTC.

    Project convention: storage is always UTC. The
    :meth:`BaseLanceTable.to_arrow_schema` rewrite ignores whatever tz a
    subclass attempts to set and replaces it with ``tz=UTC``. This pins
    that no future schema can quietly opt out of the discipline.
    """
    from typing import ClassVar as _ClassVar

    import pyarrow as pa

    from everos.core.persistence.lancedb import BaseLanceTable

    class _MisbehavingSchema(BaseLanceTable):
        TABLE_NAME: _ClassVar[str] = "_misbehaving"
        id: str
        ts: dt.datetime

        @classmethod
        def to_arrow_schema(cls):  # type: ignore[no-untyped-def]
            # Subclass tries to sneak a non-UTC tz onto the column …
            base = pa.schema(
                [
                    pa.field("id", pa.string(), nullable=False),
                    pa.field(
                        "ts", pa.timestamp("us", tz="America/New_York"), nullable=False
                    ),
                ]
            )
            # … and pipes it through BaseLanceTable's coercion. We expect
            # the coercion to override NY → UTC.
            return pa.schema(
                [
                    pa.field(f.name, pa.timestamp("us", tz="UTC"), nullable=f.nullable)
                    if pa.types.is_timestamp(f.type)
                    else f
                    for f in base
                ]
            )

    schema = _MisbehavingSchema.to_arrow_schema()
    ts_field = schema.field("ts")
    assert getattr(ts_field.type, "tz", None) == "UTC", (
        f"non-UTC subclass tz must be coerced to UTC; got {ts_field.type}"
    )


def test_lancedb_schema_auto_tags_every_datetime_field_with_tz_utc() -> None:
    """Every datetime column on a BaseLanceTable subclass gets tz=UTC auto-applied.

    Pins the **zero-configuration** contract: subclasses just declare
    ``ts: datetime`` and ``BaseLanceTable.to_arrow_schema`` rewrites
    every naive ``timestamp[us]`` to ``timestamp[us, tz=UTC]``. No
    per-table opt-in declaration is required.
    """
    import pyarrow as pa

    from everos.infra.persistence.lancedb.tables.agent_case import AgentCase
    from everos.infra.persistence.lancedb.tables.atomic_fact import AtomicFact
    from everos.infra.persistence.lancedb.tables.episode import Episode
    from everos.infra.persistence.lancedb.tables.foresight import Foresight
    from everos.infra.persistence.lancedb.tables.user_profile import UserProfile

    for cls in (Episode, AtomicFact, AgentCase, Foresight, UserProfile):
        schema = cls.to_arrow_schema()
        ts_fields = [f for f in schema if pa.types.is_timestamp(f.type)]
        assert ts_fields, f"{cls.__name__} has no timestamp fields (unexpected)"
        for field in ts_fields:
            assert getattr(field.type, "tz", None) == "UTC", (
                f"{cls.__name__}.{field.name} should be timestamp[us, tz=UTC]; "
                f"got {field.type}"
            )


def test_to_display_tz_under_default_settings_returns_z_suffix() -> None:
    """Default ``EVEROS_MEMORY__TIMEZONE=UTC`` → rendered offset is ``Z``."""
    from pydantic import BaseModel

    class _Row(BaseModel):
        ts: dt.datetime

    utc = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)
    out = to_display_tz(utc)
    rendered = _Row(ts=out).model_dump_json()
    assert '"ts":"2026-05-29T06:00:00Z"' in rendered


def test_sorting_multiple_datetimes_consistent_after_tz_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sort order of a list of UTC instants is independent of display tz.

    Display-tz conversion is a same-instant transform (astimezone is a
    bijection); sort by UTC then render must agree with sort by display-tz.
    """
    instants = [dt.datetime(2026, 5, 29, h, tzinfo=dt.UTC) for h in (8, 1, 14, 0, 23)]
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    rendered = [to_display_tz(d) for d in instants]
    sorted_via_utc = sorted(instants)
    sorted_via_rendered = sorted(rendered)
    # astimezone preserves order — pairwise alignment
    for utc_d, rendered_d in zip(sorted_via_utc, sorted_via_rendered, strict=True):
        assert utc_d == rendered_d


def test_reverse_tz_switch_utc_to_shanghai_no_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Write under UTC display, read under Shanghai display → same instant.

    Symmetric to the Shanghai→UTC drift e2e. Covers the migration scenario
    where the OG deployment defaults to UTC and a later operator turns on
    a local display tz.
    """
    import asyncio
    import json as _json

    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "UTC")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    from everos.core.persistence.sqlite import SQLModel as _SQLModel
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        sqlite_manager,
        unprocessed_buffer_repo,
    )

    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    target = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)

    async def _write_under_utc() -> None:
        engine = sqlite_manager.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(_SQLModel.metadata.create_all)
        row = UnprocessedBuffer(
            message_id="m_rev",
            session_id="s_rev",
            track="memorize",
            sender_id="alice",
            role="user",
            timestamp=target,
            content_items_json=_json.dumps([{"type": "text", "text": "x"}]),
            text="x",
        )
        await unprocessed_buffer_repo.replace("s_rev", "memorize", [row])
        await sqlite_manager.dispose_engine()

    asyncio.run(_write_under_utc())

    # Switch display tz to Shanghai, reset DB engine cache, read back.
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    sqlite_manager._engine = None
    sqlite_manager._session_factory = None

    async def _read_under_shanghai() -> None:
        rows = await unprocessed_buffer_repo.list_for_track("s_rev", "memorize")
        assert len(rows) == 1
        # storage is UTC — read back equals what we wrote
        assert rows[0].timestamp == target
        # display-tz render shifts wall-clock by +08:00 without changing instant
        rendered = to_display_tz(rows[0].timestamp)
        assert rendered.hour == 14
        assert rendered.utcoffset() == dt.timedelta(hours=8)
        await sqlite_manager.dispose_engine()

    asyncio.run(_read_under_shanghai())


def test_from_timestamp_ms_round_trip_through_ensure_utc() -> None:
    """ms epoch → from_timestamp → ensure_utc must preserve the UTC instant.

    The ``/add`` request body declares timestamps as Unix epoch ms; this
    test pins the conversion chain from wire format to storage.
    """
    ms = 1748498400000  # 2026-05-29T06:00:00Z
    via_helper = from_timestamp(ms)
    via_utc = ensure_utc(via_helper)
    assert via_utc is not None
    assert via_utc.tzinfo is dt.UTC
    assert int(via_utc.timestamp() * 1000) == ms


def test_sqlite_before_insert_event_normalises_aware_non_utc_to_utc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """SQLAlchemy ``before_insert`` mapper event converts aware Shanghai → UTC.

    Pins the **write-side** half of the storage-UTC discipline.
    ``SQLModel(table=True)`` classes skip Pydantic ``AfterValidator``,
    so the :data:`UtcDatetime` annotation by itself is **inert** at
    construction. The mapper event registered in
    :mod:`everos.core.persistence.sqlite.base` is what guarantees the
    on-disk SQLite text is UTC bytes, not display-tz bytes.

    Test path: write a row whose ``timestamp`` is aware Shanghai 14:00,
    then probe SQLite with a raw SQL ``SELECT`` (bypassing the load hook
    so we observe what's literally on disk).
    """
    import asyncio
    import json as _json

    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    from zoneinfo import ZoneInfo

    from sqlalchemy import text as _sql_text

    from everos.core.persistence.sqlite import SQLModel as _SQLModel
    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import (
        UnprocessedBuffer,
        get_session_factory,
        sqlite_manager,
    )

    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    aware_sh = dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("Asia/Shanghai"))

    async def _run() -> None:
        engine = sqlite_manager.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(_SQLModel.metadata.create_all)

        row = UnprocessedBuffer(
            message_id="m1",
            session_id="s1",
            track="memorize",
            sender_id="alice",
            role="user",
            timestamp=aware_sh,
            content_items_json=_json.dumps([{"type": "text", "text": "x"}]),
            text="x",
        )
        # Sanity: table=True SQLModel skips Pydantic validators, so the
        # construction site does NOT normalise the timestamp. The event
        # listener is what does it later, at write time.
        assert row.timestamp == aware_sh, (
            "test invariant: construction did NOT normalise"
        )

        async with session_scope(get_session_factory()) as session:
            session.add(row)
            await session.commit()

        # Probe raw SQLite — bypass the load hook by issuing raw SQL.
        async with engine.connect() as conn:
            raw = (
                await conn.execute(
                    _sql_text(
                        "SELECT timestamp FROM unprocessed_buffer WHERE message_id='m1'"
                    )
                )
            ).scalar()

        # Aware Shanghai 14:00 = UTC 06:00. The on-disk bytes should be
        # the UTC wall-clock, not Shanghai's.
        assert "06:00:00" in raw, (
            f"on-disk should be UTC 06:00:00, not Shanghai 14:00:00; got {raw!r}"
        )
        assert "14:00:00" not in raw

        await sqlite_manager.dispose_engine()

    asyncio.run(_run())


# ── None-passthrough boundary (Gap #1) ───────────────────────────────────


def test_ensure_utc_returns_none_for_none() -> None:
    """``ensure_utc(None)`` is a no-op — supports nullable repo columns directly."""
    assert ensure_utc(None) is None


def test_to_display_tz_returns_none_for_none() -> None:
    """``to_display_tz(None)`` is a no-op — supports nullable repo columns directly."""
    assert to_display_tz(None) is None


def test_ensure_utc_and_display_tz_chained_through_none() -> None:
    """``to_display_tz(ensure_utc(None))`` short-circuits without ``AttributeError``.

    Pins the common shaper pattern against nullable storage columns like
    ``MdChangeState.last_attempt_at``.
    """
    assert to_display_tz(ensure_utc(None)) is None


# ── SQLite load-event hook cross-table (Gap #2) ──────────────────────────


def test_sqlite_load_hook_attaches_utc_on_all_base_table_subclasses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Every ``BaseTable`` subclass with a ``UtcDatetime`` column gets UTC on read.

    Pins the centralised defense: the SQLAlchemy ``load`` event hook
    on ``BaseTable`` works for *every* subclass, not just the one we
    happened to test. Inserts a row in each real table carrying a known
    UTC instant, reads back via the repo / a plain ``select``, then
    asserts ``tzinfo is UTC`` and value preservation across:

    - ``BaseTable.created_at`` / ``updated_at`` on every subclass
    - per-table business datetime columns
      (``timestamp`` / ``last_message_ts`` / ``last_memcell_ts`` /
       ``first_seen_at`` / ``last_changed_at`` / ``last_attempt_at``).
    """
    import asyncio
    import json as _json

    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    from sqlmodel import select

    from everos.core.persistence.sqlite import SQLModel as _SQLModel
    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import (
        ConversationStatus,
        MdChangeState,
        Memcell,
        UnprocessedBuffer,
        get_session_factory,
        sqlite_manager,
    )

    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    target = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)

    async def _run() -> None:
        engine = sqlite_manager.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(_SQLModel.metadata.create_all)

        rows = [
            UnprocessedBuffer(
                message_id="m1",
                session_id="s1",
                track="memorize",
                sender_id="alice",
                role="user",
                timestamp=target,
                content_items_json=_json.dumps([{"type": "text", "text": "x"}]),
                text="x",
            ),
            Memcell(
                memcell_id="mc1",
                session_id="s1",
                track="memorize",
                raw_type="Conversation",
                message_ids_json=_json.dumps(["m1"]),
                sender_ids_json=_json.dumps(["alice"]),
                payload_json="{}",
                timestamp=target,
            ),
            ConversationStatus(
                session_id="s1",
                track="memorize",
                last_message_ts=target,
                last_memcell_ts=target,
            ),
            MdChangeState(
                md_path="users/alice/episodes/episode-2026-05-29.md",
                kind="episode",
                change_type="added",
                mtime=0.0,
                lsn=1,
                last_attempt_at=target,
            ),
        ]

        async with session_scope(get_session_factory()) as session:
            for row in rows:
                session.add(row)
            await session.commit()

        async with session_scope(get_session_factory()) as session:
            ub = (await session.execute(select(UnprocessedBuffer))).scalar_one()
            mc = (await session.execute(select(Memcell))).scalar_one()
            cs = (await session.execute(select(ConversationStatus))).scalar_one()
            mcs = (await session.execute(select(MdChangeState))).scalar_one()

        # BaseTable's created_at / updated_at on every row
        for row, name in [
            (ub, "UnprocessedBuffer"),
            (mc, "Memcell"),
            (cs, "ConversationStatus"),
            (mcs, "MdChangeState"),
        ]:
            assert row.created_at.tzinfo is dt.UTC, (
                f"{name}.created_at not aware UTC; got {row.created_at.tzinfo!r}"
            )
            assert row.updated_at.tzinfo is dt.UTC, (
                f"{name}.updated_at not aware UTC; got {row.updated_at.tzinfo!r}"
            )

        # Per-table business datetime columns
        assert ub.timestamp.tzinfo is dt.UTC and ub.timestamp == target
        assert mc.timestamp.tzinfo is dt.UTC and mc.timestamp == target
        assert cs.last_message_ts is not None
        assert cs.last_message_ts.tzinfo is dt.UTC
        assert cs.last_message_ts == target
        assert cs.last_memcell_ts is not None
        assert cs.last_memcell_ts.tzinfo is dt.UTC
        assert cs.last_memcell_ts == target
        assert mcs.first_seen_at.tzinfo is dt.UTC
        assert mcs.last_changed_at.tzinfo is dt.UTC
        assert mcs.last_attempt_at is not None
        assert mcs.last_attempt_at.tzinfo is dt.UTC
        assert mcs.last_attempt_at == target

        await sqlite_manager.dispose_engine()

    asyncio.run(_run())


# ── SQLAlchemy write-path coverage (TypeDecorator) ───────────────────────
#
# The previous defense relied on mapper events (``before_insert`` /
# ``before_update``), which ONLY fire on the ORM unit-of-work flush
# path. Core SQL statements (``session.execute(insert(...).values())``,
# ``update(...).values()``, ``delete(...)``, bulk operations) bypass
# them — and md_change_state_repo uses Core statements pervasively.
#
# The fix is :class:`UtcDateTimeColumn`, a SQLAlchemy ``TypeDecorator``
# whose ``process_bind_param`` runs on **every** bind parameter
# regardless of the calling API. These tests pin every write path
# against the storage-UTC contract.


def _build_engine_for_test(monkeypatch, tmp_path, tz: str = "Asia/Shanghai"):
    """Common setup: tmp memory root + tz + fresh engine."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", tz)
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()

    from everos.infra.persistence.sqlite import sqlite_manager

    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    return sqlite_manager


async def _create_schema(sqlite_manager) -> None:
    from everos.core.persistence.sqlite import SQLModel as _SQLModel

    engine = sqlite_manager.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(_SQLModel.metadata.create_all)


async def _probe_raw_text(sqlite_manager, sql: str) -> str | None:
    """Read a single column via raw SQL — bypasses ORM hydrate hooks."""
    from sqlalchemy import text as _sql_text

    engine = sqlite_manager.get_engine()
    async with engine.connect() as conn:
        return (await conn.execute(_sql_text(sql))).scalar()


def test_typedec_covers_orm_session_add(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """ORM ``session.add`` write path: aware Shanghai → UTC bytes on disk."""
    import asyncio
    from zoneinfo import ZoneInfo

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path)
    aware_sh = dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("Asia/Shanghai"))

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            s.add(
                MdChangeState(
                    md_path="p_orm",
                    kind="ep",
                    change_type="added",
                    mtime=0.0,
                    lsn=1,
                    last_attempt_at=aware_sh,
                )
            )
            await s.commit()
        raw = await _probe_raw_text(
            sm,
            "SELECT last_attempt_at FROM md_change_state WHERE md_path='p_orm'",
        )
        assert raw and "06:00" in raw and "14:00" not in raw
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_covers_core_insert_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Core ``insert(Model).values(...)`` bypasses ORM but TypeDecorator catches it."""
    import asyncio
    from zoneinfo import ZoneInfo

    from sqlalchemy import insert

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path)
    aware_sh = dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("Asia/Shanghai"))

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            await s.execute(
                insert(MdChangeState).values(
                    md_path="p_core_ins",
                    kind="ep",
                    change_type="added",
                    mtime=0.0,
                    lsn=2,
                    last_attempt_at=aware_sh,
                )
            )
            await s.commit()
        raw = await _probe_raw_text(
            sm,
            "SELECT last_attempt_at FROM md_change_state WHERE md_path='p_core_ins'",
        )
        assert raw and "06:00" in raw and "14:00" not in raw
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_covers_core_update_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Core ``update(Model).where(...).values(...)`` — the path
    md_change_state_repo uses pervasively. TypeDecorator must catch it.
    """
    import asyncio
    from zoneinfo import ZoneInfo

    from sqlalchemy import update

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path)
    aware_sh = dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("Asia/Shanghai"))

    async def _run() -> None:
        await _create_schema(sm)
        # Seed a row first
        async with session_scope(get_session_factory()) as s:
            s.add(
                MdChangeState(
                    md_path="p_upd",
                    kind="ep",
                    change_type="added",
                    mtime=0.0,
                    lsn=3,
                )
            )
            await s.commit()
        # Now Core update with aware non-UTC datetime
        async with session_scope(get_session_factory()) as s:
            await s.execute(
                update(MdChangeState)
                .where(MdChangeState.md_path == "p_upd")
                .values(last_attempt_at=aware_sh)
            )
            await s.commit()
        raw = await _probe_raw_text(
            sm,
            "SELECT last_attempt_at FROM md_change_state WHERE md_path='p_upd'",
        )
        assert raw and "06:00" in raw and "14:00" not in raw
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_aware_utc_input_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gap a: aware UTC input round-trips unchanged."""
    import asyncio

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path)
    aware_utc = dt.datetime(2026, 5, 29, 6, tzinfo=dt.UTC)

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            s.add(
                MdChangeState(
                    md_path="p_utc",
                    kind="ep",
                    change_type="added",
                    mtime=0.0,
                    lsn=4,
                    last_attempt_at=aware_utc,
                )
            )
            await s.commit()
        # Raw bytes
        raw = await _probe_raw_text(
            sm,
            "SELECT last_attempt_at FROM md_change_state WHERE md_path='p_utc'",
        )
        assert raw and "06:00" in raw
        # Read-back
        from sqlmodel import select

        async with session_scope(get_session_factory()) as s:
            row = (
                await s.execute(
                    select(MdChangeState).where(MdChangeState.md_path == "p_utc")
                )
            ).scalar_one()
        assert row.last_attempt_at == aware_utc
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_naive_input_treated_as_utc(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gap b: naive datetime input → assumed already-UTC, NOT display-tz.

    Even with display tz = Shanghai, a naive 14:00 input is stored as
    14:00 UTC (not interpreted as Shanghai 14:00 = UTC 06:00). This
    pins the project's "storage convention: naive = UTC" rule.
    """
    import asyncio

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path)
    naive = dt.datetime(2026, 5, 29, 14)  # no tzinfo

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            s.add(
                MdChangeState(
                    md_path="p_naive",
                    kind="ep",
                    change_type="added",
                    mtime=0.0,
                    lsn=5,
                    last_attempt_at=naive,
                )
            )
            await s.commit()
        raw = await _probe_raw_text(
            sm,
            "SELECT last_attempt_at FROM md_change_state WHERE md_path='p_naive'",
        )
        # naive 14:00 → stored 14:00 (assumed UTC), NOT 06:00 (which would
        # mean we re-interpreted naive as Shanghai-local)
        assert raw and "14:00" in raw, f"naive should land as 14:00 UTC; got {raw!r}"
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_microsecond_precision_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gap d: microsecond field survives the round trip."""
    import asyncio

    from sqlmodel import select

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path, tz="UTC")
    with_micros = dt.datetime(2026, 5, 29, 6, 0, 0, 123_456, tzinfo=dt.UTC)

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            s.add(
                MdChangeState(
                    md_path="p_us",
                    kind="ep",
                    change_type="added",
                    mtime=0.0,
                    lsn=6,
                    last_attempt_at=with_micros,
                )
            )
            await s.commit()
        async with session_scope(get_session_factory()) as s:
            row = (
                await s.execute(
                    select(MdChangeState).where(MdChangeState.md_path == "p_us")
                )
            ).scalar_one()
        assert row.last_attempt_at.microsecond == 123_456
        assert row.last_attempt_at == with_micros
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_extreme_dates_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gap f: 1970 and 2099 epoch endpoints round-trip without overflow."""
    import asyncio

    from sqlmodel import select

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path, tz="UTC")
    epoch_start = dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
    far_future = dt.datetime(2099, 12, 31, 23, 59, 59, tzinfo=dt.UTC)

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            s.add_all(
                [
                    MdChangeState(
                        md_path="p_1970",
                        kind="ep",
                        change_type="added",
                        mtime=0.0,
                        lsn=7,
                        last_attempt_at=epoch_start,
                    ),
                    MdChangeState(
                        md_path="p_2099",
                        kind="ep",
                        change_type="added",
                        mtime=0.0,
                        lsn=8,
                        last_attempt_at=far_future,
                    ),
                ]
            )
            await s.commit()
        async with session_scope(get_session_factory()) as s:
            r1 = (
                await s.execute(
                    select(MdChangeState).where(MdChangeState.md_path == "p_1970")
                )
            ).scalar_one()
            r2 = (
                await s.execute(
                    select(MdChangeState).where(MdChangeState.md_path == "p_2099")
                )
            ).scalar_one()
        assert r1.last_attempt_at == epoch_start
        assert r2.last_attempt_at == far_future
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_dst_boundary_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gap g: a Shanghai-input instant that straddles US DST boundary preserves UTC.

    A 14:00 +08:00 on 2026-03-08 is the same UTC instant whether read
    in pre-DST or post-DST US tz. The TypeDecorator must not introduce
    DST artefacts when astimezone-ing.
    """
    import asyncio
    from zoneinfo import ZoneInfo

    from sqlmodel import select

    from everos.core.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite import MdChangeState, get_session_factory

    sm = _build_engine_for_test(monkeypatch, tmp_path, tz="Asia/Shanghai")
    # US DST starts 2026-03-08 (2am local → 3am local). Pick an instant
    # straddling the boundary in NY tz.
    pre_dst = dt.datetime(2026, 3, 8, 6, 30, tzinfo=ZoneInfo("America/New_York"))
    post_dst = dt.datetime(2026, 3, 8, 7, 30, tzinfo=ZoneInfo("America/New_York"))

    async def _run() -> None:
        await _create_schema(sm)
        async with session_scope(get_session_factory()) as s:
            s.add_all(
                [
                    MdChangeState(
                        md_path="p_pre",
                        kind="ep",
                        change_type="added",
                        mtime=0.0,
                        lsn=9,
                        last_attempt_at=pre_dst,
                    ),
                    MdChangeState(
                        md_path="p_post",
                        kind="ep",
                        change_type="added",
                        mtime=0.0,
                        lsn=10,
                        last_attempt_at=post_dst,
                    ),
                ]
            )
            await s.commit()
        async with session_scope(get_session_factory()) as s:
            r1 = (
                await s.execute(
                    select(MdChangeState).where(MdChangeState.md_path == "p_pre")
                )
            ).scalar_one()
            r2 = (
                await s.execute(
                    select(MdChangeState).where(MdChangeState.md_path == "p_post")
                )
            ).scalar_one()
        # Both must round-trip exactly. The UTC instant is invariant under
        # any tz transformation including DST shifts.
        assert r1.last_attempt_at == pre_dst
        assert r2.last_attempt_at == post_dst
        await sm.dispose_engine()

    asyncio.run(_run())


def test_typedec_raw_sql_bypasses_typedecorator_documented_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gap h: pin the **known limit**: pure raw SQL with literal strings
    bypasses the column type entirely. If a future contributor writes
    ``text("INSERT ... VALUES ('14:00:00')")``, they get the bytes they
    typed — no normalisation.

    This test documents the limit so it does not regress to silent
    "we thought TypeDecorator covered everything". The real defense
    against raw SQL is the ``check_datetime_discipline.py`` scanner.
    """
    import asyncio

    from sqlalchemy import text as _sql_text

    sm = _build_engine_for_test(monkeypatch, tmp_path, tz="Asia/Shanghai")

    async def _run() -> None:
        await _create_schema(sm)
        # Raw SQL with a literal Shanghai 14:00 string — no bind param
        # goes through TypeDecorator, so the literal lands as-is.
        engine = sm.get_engine()
        async with engine.begin() as conn:
            await conn.execute(
                _sql_text(
                    "INSERT INTO md_change_state "
                    "(md_path, kind, change_type, mtime, "
                    "first_seen_at, last_changed_at, lsn, status, "
                    "last_attempt_at, retry_count, "
                    "created_at, updated_at) "
                    "VALUES "
                    "('p_raw', 'ep', 'added', 0.0, "
                    "'2026-05-29 14:00:00', '2026-05-29 14:00:00', 99, "
                    "'pending', '2026-05-29 14:00:00', 0, "
                    "'2026-05-29 14:00:00', '2026-05-29 14:00:00')"
                )
            )

        raw = await _probe_raw_text(
            sm,
            "SELECT last_attempt_at FROM md_change_state WHERE md_path='p_raw'",
        )
        # Confirms the LIMIT: raw literal stored as-is, no normalisation.
        assert "14:00" in raw, (
            "Raw SQL with literal datetime string is NOT normalised by "
            "TypeDecorator. This is a documented limit; "
            "scripts/check_datetime_discipline.py forbids new bypasses."
        )
        await sm.dispose_engine()

    asyncio.run(_run())


# ── LanceDB write-path coverage ──────────────────────────────────────────
#
# LanceDB has fewer write APIs: ``table.add`` (the main one),
# ``table.merge_insert``, and ``table.update``. All of them ultimately go
# through PyArrow which uses the Arrow schema to coerce input. The
# ``BaseLanceTable.to_arrow_schema`` rewrite that stamps ``tz=UTC`` is
# therefore active on every write path. Pin this explicitly.


def test_lance_table_add_normalises_aware_non_utc() -> None:
    """LanceDB ``table.add`` with aware Shanghai input → aware UTC on disk."""
    import asyncio
    import tempfile
    from zoneinfo import ZoneInfo

    import lancedb

    from everos.infra.persistence.lancedb.tables.episode import Episode

    aware_sh = dt.datetime(2026, 5, 29, 14, tzinfo=ZoneInfo("Asia/Shanghai"))

    async def _run() -> None:
        conn = await lancedb.connect_async(tempfile.mkdtemp())
        table = await conn.create_table("ep", schema=Episode)
        row = Episode(
            id="alice_ep_1",
            entry_id="ep_1",
            owner_id="alice",
            owner_type="user",
            session_id="s1",
            timestamp=aware_sh,
            parent_id="mc_1",
            sender_ids=["alice"],
            episode="x",
            episode_tokens="x",
            md_path="users/alice/episodes/x.md",
            content_sha256="abc",
            vector=[0.0] * 1024,
        )
        await table.add([row])
        rows = await table.query().to_list()
        assert rows[0]["timestamp"].tzinfo is not None
        # 14:00 +08:00 = 06:00 UTC
        assert rows[0]["timestamp"].hour == 6
        assert rows[0]["timestamp"] == aware_sh

    asyncio.run(_run())


def test_lance_table_naive_input_is_assumed_utc() -> None:
    """LanceDB naive datetime → PyArrow assumes UTC (matches project convention)."""
    import asyncio
    import tempfile

    import lancedb

    from everos.infra.persistence.lancedb.tables.episode import Episode

    naive = dt.datetime(2026, 5, 29, 14)  # naive — no tz

    async def _run() -> None:
        conn = await lancedb.connect_async(tempfile.mkdtemp())
        table = await conn.create_table("ep", schema=Episode)
        row = Episode(
            id="x_1",
            entry_id="ep_1",
            owner_id="x",
            owner_type="user",
            session_id="s",
            timestamp=naive,
            parent_id="mc",
            sender_ids=[],
            episode="x",
            episode_tokens="x",
            md_path="x",
            content_sha256="x",
            vector=[0.0] * 1024,
        )
        await table.add([row])
        rows = await table.query().to_list()
        # naive 14:00 → assumed UTC 14:00 on disk → read back aware UTC 14:00
        assert rows[0]["timestamp"].hour == 14
        assert rows[0]["timestamp"].tzinfo is not None
        assert rows[0]["timestamp"].utcoffset() == dt.timedelta(0)

    asyncio.run(_run())
