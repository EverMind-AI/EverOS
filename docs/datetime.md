# Datetime & Timezones

> Audience: contributors. Read this once before touching any code that
> records a moment in time.

## Table of contents

- [The two-zone discipline](#the-two-zone-discipline)
- [Why two zones](#why-two-zones)
- [Helper reference](#helper-reference)
- [Field-type rules](#field-type-rules)
- [End-to-end data flow](#end-to-end-data-flow)
- [Common pitfalls](#common-pitfalls)
- [Testing guidance](#testing-guidance)

## The two-zone discipline

EverOS treats datetimes on **two separate rails**:

| Rail | Where it lives | Helper |
|---|---|---|
| **UTC** (storage) | SQLite, LanceDB, OME events — anything persisted to disk | `get_utc_now`, `ensure_utc`, `UtcDatetime` |
| **Display tz** | Markdown frontmatter, HTTP API responses, daily-log filename buckets, fallback zone for naive caller input | `get_now_with_timezone`, `today_with_timezone`, `to_display_tz` |

The display timezone is set by the `EVEROS_MEMORY__TIMEZONE`
environment variable (or `[memory] timezone` in TOML). Default `UTC`.

**Inviolable rule**: the display tz must **never** reach storage. Once
the user switches `EVEROS_MEMORY__TIMEZONE`, existing on-disk rows
must not misalign.

## Why two zones

### What goes wrong with a single "configured" zone

The naive design — "use one configured timezone everywhere" — has two
failure modes, both subtle:

1. **Configuration drift.** Day 1 the user configures
   `EVEROS_MEMORY__TIMEZONE=Asia/Shanghai`. Everything stores
   Shanghai-local datetimes. On Day 30 they switch to
   `UTC`. SQLite (which strips tz on write and returns naive on read)
   silently reinterprets the old Shanghai values as UTC — every old
   row jumps eight hours into the future.
2. **Cross-region replication.** If two deployments share storage
   but configure different display zones, both interpret the same
   naive bytes against their own local zone and diverge by the
   offset delta. There is no "true" reading.

UTC-only storage forecloses both: bytes on disk are zone-independent.

### Why not UTC everywhere then?

Users want to read timestamps in their wall-clock zone. Markdown
frontmatter that says `2026-05-29T06:00:00Z` for a meeting that
happened locally at 14:00 is jarring. The display rail solves this
without polluting storage: render UTC bytes through `to_display_tz`
at the boundary.

## Helper reference

All helpers live in [`everos.component.utils.datetime`](../src/everos/component/utils/datetime.py).

### Storage rail

| Helper | Behaviour |
|---|---|
| `get_utc_now() -> datetime` | Current UTC instant, `tzinfo=UTC`. Independent of any setting. Use as `default_factory` on any storage field. |
| `ensure_utc(d) -> datetime` | Naive → attach display tz → convert to UTC. Aware → `astimezone(UTC)`. Use at the storage boundary if you receive a datetime you didn't construct. |
| `UtcDatetime` | `Annotated[datetime, AfterValidator(ensure_utc)]`. Apply to any SQLite field. Pydantic auto-runs validation on both INSERT defaults and read-back rows. |

### Display rail

| Helper | Behaviour |
|---|---|
| `get_now_with_timezone() -> datetime` | Current instant in the configured display tz. `.isoformat()` produces e.g. `2026-05-29T14:00:00+08:00`. |
| `today_with_timezone() -> date` | Today's date in the display tz. Use for daily-log filename buckets. |
| `to_display_tz(d) -> datetime` | Convert any datetime to the display tz. Naive input is treated as already display-tz local. |

### Parsing & rendering

| Helper | Behaviour |
|---|---|
| `from_iso_format(value)` | Parse an ISO string / datetime / epoch. Naive input attaches **display tz** (the "if you didn't say a zone, assume your zone" rule). |
| `from_timestamp(ts)` | Parse epoch seconds / milliseconds (auto-detects). Returns display-tz aware. |
| `to_iso_format(d)` | `.isoformat()` after light validation. |
| `to_timestamp_ms(d)` | Milliseconds epoch (`int`). |

## Field-type rules

### SQLite tables

```python
from everos.component.utils.datetime import UtcDatetime, get_utc_now
from everos.core.persistence.sqlite import BaseTable, Field

class MyRow(BaseTable, table=True):
    happened_at: UtcDatetime = Field(default_factory=get_utc_now)
```

Why `UtcDatetime` and not plain `datetime`? SQLAlchemy silently strips
tz on SQLite writes. `UtcDatetime`'s `AfterValidator` runs on
**construction** to make sure whatever the caller hands in gets
normalised to UTC before persistence.

SQLModel's ORM hydrate path (rows from `select(...)`) **bypasses**
the Pydantic validator — SQLAlchemy assigns column values straight
to instance attributes. To close that gap,
[core/persistence/sqlite/base.py](../src/everos/core/persistence/sqlite/base.py)
registers a SQLAlchemy `load` event listener that re-attaches
`tzinfo=UTC` to every `UtcDatetime` column after hydrate. Net effect:
**callers never see a naive datetime from a SQLite repo**, whatever
the code path.

`BaseTable.created_at` / `updated_at` already use `UtcDatetime` and
`get_utc_now` — any subclass inherits both the construction-time
validator **and** the load-time hook for free.

### LanceDB tables — zero configuration

```python
import datetime as _dt

class MyLanceRow(BaseLanceTable):
    ts: _dt.datetime   # automatically tz=UTC in the Arrow schema
```

LanceDB's Pydantic → PyArrow converter does not understand
`typing.Annotated` metadata; using `UtcDatetime` as the annotation
would raise `TypeError: Converting Pydantic type to Arrow Type`.
Instead, `BaseLanceTable.to_arrow_schema()` walks the inferred schema
and rewrites **every** naive `timestamp[us]` column to
`timestamp[us, tz=UTC]`. PyArrow then:

* **on write** — `astimezone(UTC)` any aware input automatically.
* **on read** — returns aware UTC datetimes (not naive).

No caller-side coercion needed, no per-table declaration. The
response shapers only run `to_display_tz(...)` to convert UTC to the
configured display zone.

If a future schema genuinely needs a naive datetime column (project
convention says storage is always UTC, so this would be unusual),
override `to_arrow_schema` on that subclass and skip the patch for
that one column.

### OME events / in-memory state

OME events are persisted-adjacent (the `run_record` / `counter` stores
serialise them). Use `get_utc_now()` for any `default_factory` on the
event payload.

## Two centralised defenses

| Backend | Defense | Where |
|---|---|---|
| **SQLite** | SQLAlchemy `load` event listener on `BaseTable` re-attaches `tzinfo=UTC` after every ORM hydrate | [core/persistence/sqlite/base.py](../src/everos/core/persistence/sqlite/base.py) |
| **LanceDB** | `BaseLanceTable.to_arrow_schema()` rewrites `UTC_DATETIME_FIELDS` columns to `timestamp[us, tz=UTC]`; PyArrow handles UTC end-to-end | [core/persistence/lancedb/base.py](../src/everos/core/persistence/lancedb/base.py) |
| **CI gate** | `scripts/check_datetime_discipline.py` fails the build on any code that bypasses `component/utils/datetime` | wired into `make lint` |

These defenses replace what used to be an "every consumer must call
`ensure_utc()`" shotgun discipline. With both in place, callers never
observe a naive datetime from either backend.

## End-to-end data flow

```
User input (any zone)
        │
        ▼
   from_iso_format     ←  naive → attach display tz
        │
        ▼
   ensure_utc          ←  storage boundary: → UTC
        │
        ▼
┌────────────────┬────────────────┐
│   SQLite       │   LanceDB      │
│  (UtcDatetime  │   (Arrow       │
│   re-attaches  │   stripped to  │
│   UTC on read) │   UTC bytes)   │
└────────────────┴────────────────┘
        │
        ▼
   from_iso_format    ←  read path normalises naive → display tz
        │
        ▼
   to_display_tz      ←  response boundary: → display tz
        │
        ▼
   Pydantic .isoformat()  →  "2026-05-29T14:00:00+08:00"
        │
        ▼
   HTTP API response / markdown frontmatter
```

The storage boundary and response boundary are the two points where
the zone discipline is enforced. Everything in between just passes
datetimes through.

## Common pitfalls

> [!WARNING]
> **`datetime.now()` without `tz=`.** Forbidden. Always use
> `get_utc_now()` (storage) or `get_now_with_timezone()` (display).
> Linted by `.claude/rules/datetime-handling.md` and CI.

> [!WARNING]
> **Calling `astimezone()` on a value just read from SQLite.** If the
> field isn't typed `UtcDatetime`, SQLite returns naive — and
> `astimezone()` on a naive datetime silently interprets it as
> **local process time**, not UTC. Always use `UtcDatetime` on SQLite
> fields.

> [!WARNING]
> **Storing `get_now_with_timezone()` directly.** That returns
> display-tz time. If the display tz later changes, your stored values
> are stranded. Use `get_utc_now()` for any persisted field.

> [!INFO]
> **Migrating existing rows.** Q2 was rolled out on a clean codebase
> with no production data. If you operate an instance where SQLite
> values were written with display-tz-aware values (pre-Q2), you must
> either drop the database or write a one-time migration that
> reinterprets each row's naive value against the old display tz
> before re-writing as UTC. The project does not ship such a
> migration.

## Testing guidance

For unit tests that depend on display-tz behaviour, both caches must
clear:

```python
import pytest
from everos.component.utils import datetime as dt_module
from everos.config import load_settings

@pytest.fixture(autouse=True)
def _isolate_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVEROS_MEMORY__TIMEZONE", raising=False)
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
```

The autouse fixture in [tests/conftest.py](../tests/conftest.py) does
exactly this — it runs for every test by default. If you write a
locally-scoped test that needs a non-default zone, monkeypatch the env
var **and** clear both caches:

```python
def test_my_thing(monkeypatch):
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    ...
```

The full invariant set is covered in
[tests/unit/test_component/test_utils/test_datetime.py](../tests/unit/test_component/test_utils/test_datetime.py)
under the "Q2 two-zone discipline invariants" section. If you change
the storage / display contract, those tests are the first line of
defense — update them in lockstep.
