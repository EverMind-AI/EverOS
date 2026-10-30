---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Datetime handling rule (two-zone discipline)

**Never** construct or read "now" directly. All datetime flows through
`everos.core.utils.datetime`. This is a **hard CI gate**
(`make check-datetime`, wired into `make lint`).

## Banned (the checker fails the build on these)

- `datetime.now()`, `datetime.utcnow()`, `datetime.today()`
- `time.time()`, `time.time_ns()`
- `datetime(YYYY, ...)` without `tzinfo=`
- `.astimezone(...)` / `.replace(tzinfo=...)` outside the helper module

## Use instead

| Need | Helper |
|---|---|
| "now" for **storage** (UTC) | `get_utc_now()` |
| "now" for **display** (configured TZ) | `get_now_with_timezone()` |
| today's date, display TZ | `today_with_timezone()` |
| normalize a value to UTC | `ensure_utc(d)` |
| render to display TZ | `to_display_tz(d)` |
| parse ISO / epoch / str | `from_iso_format(v)`, `from_timestamp(ts)` |
| serialize | `to_iso_format(d)`, `to_date_str(d)`, `to_timestamp_ms(d)` |

**Two zones**: persist in UTC, present in the configured display TZ. Crossing them
goes through the helpers — never ad-hoc. See [docs/datetime.md](../../docs/datetime.md).
