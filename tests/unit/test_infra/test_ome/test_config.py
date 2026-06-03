from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.infra.ome.config import (
    CounterOverride,
    OMEConfig,
    StrategyOverride,
    TomlRoot,
)


def test_ome_config_defaults() -> None:
    from everos.core.persistence.memory_root import MemoryRoot

    c = OMEConfig()
    assert c.jobstore_path == MemoryRoot.default().ome_db
    assert c.aps_jobstore_path == MemoryRoot.default().ome_aps_db
    assert c.max_concurrent_runs == 20
    assert c.max_retries == 1
    assert c.max_records_per_strategy == 1000
    assert c.crash_recovery_timeout_seconds == 1800
    assert c.config_path is None
    assert c.config_watch is True
    assert c.config_watch_debounce_ms == 1600


def test_aps_jobstore_path_derives_sibling_of_jobstore_path(tmp_path: object) -> None:
    """When only ``jobstore_path`` is set, APS db lands next to it as
    ``<stem>.aps.db`` so callers using a custom path (e.g. tests with
    tmp_path) get an isolated APS file rather than the global default."""
    from pathlib import Path

    custom = Path(str(tmp_path)) / "custom_dir" / "my_ome.db"
    c = OMEConfig(jobstore_path=custom)
    assert c.aps_jobstore_path == custom.with_name("my_ome.aps.db")


def test_aps_jobstore_path_respects_explicit_value(tmp_path: object) -> None:
    """An explicitly passed ``aps_jobstore_path`` is honored verbatim and
    the derivation validator does not overwrite it."""
    from pathlib import Path

    ome = Path(str(tmp_path)) / "ome.db"
    aps = Path(str(tmp_path)) / "elsewhere" / "scheduler.db"
    c = OMEConfig(jobstore_path=ome, aps_jobstore_path=aps)
    assert c.aps_jobstore_path == aps


def test_ome_config_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        OMEConfig(unknown_field=1)  # type: ignore[call-arg]


def test_ome_config_rejects_zero_concurrency() -> None:
    with pytest.raises(ValidationError):
        OMEConfig(max_concurrent_runs=0)


def test_toml_root_parses_strategy_override() -> None:
    raw = """
[strategies.cluster_memcells]
enabled = true
max_retries = 3

[strategies.cluster_memcells.gate]
threshold = 10
event_field = "user_id"
"""
    import tomllib

    parsed = tomllib.loads(raw)
    root = TomlRoot.model_validate(parsed)
    s = root.strategies["cluster_memcells"]
    assert isinstance(s, StrategyOverride)
    assert s.enabled is True
    assert s.max_retries == 3
    assert isinstance(s.gate, CounterOverride)
    assert s.gate.threshold == 10
    assert s.gate.event_field == "user_id"


def test_toml_root_forbids_unknown_strategy_field() -> None:
    import tomllib

    raw = """
[strategies.x]
unknown_key = 1
"""
    parsed = tomllib.loads(raw)
    with pytest.raises(ValidationError):
        TomlRoot.model_validate(parsed)


def test_strategy_override_accepts_cron_field() -> None:
    s = StrategyOverride(cron="0 3 * * *")
    assert s.cron == "0 3 * * *"


def test_strategy_override_accepts_idle_seconds() -> None:
    s = StrategyOverride(idle_seconds=30)
    assert s.idle_seconds == 30


def test_strategy_override_accepts_scan_interval_seconds() -> None:
    s = StrategyOverride(scan_interval_seconds=15)
    assert s.scan_interval_seconds == 15


def test_strategy_override_rejects_zero_idle_seconds() -> None:
    with pytest.raises(ValidationError):
        StrategyOverride(idle_seconds=0)


def test_strategy_override_rejects_zero_scan_interval() -> None:
    with pytest.raises(ValidationError):
        StrategyOverride(scan_interval_seconds=0)


def test_strategy_override_defaults_are_none() -> None:
    s = StrategyOverride()
    assert s.cron is None
    assert s.idle_seconds is None
    assert s.scan_interval_seconds is None


def test_counter_override_rejects_empty_event_field() -> None:
    with pytest.raises(ValidationError, match="event_field"):
        CounterOverride(event_field="")


def test_strategy_override_rejects_invalid_cron_at_construction() -> None:
    """cron is parsed by APS at construction time so TOML reload can't
    bring an invalid crontab into the system."""
    with pytest.raises(ValidationError, match="cron"):
        StrategyOverride(cron="not a cron")


def test_strategy_override_rejects_inconsistent_idle_pair() -> None:
    """When both idle_seconds and scan_interval_seconds are overridden in
    the same payload, scan_interval must be <= idle_seconds // 2 — mirror
    of the Idle trigger constraint."""
    with pytest.raises(ValidationError, match="scan_interval_seconds"):
        StrategyOverride(idle_seconds=30, scan_interval_seconds=20)


def test_strategy_override_accepts_consistent_idle_pair() -> None:
    s = StrategyOverride(idle_seconds=60, scan_interval_seconds=30)
    assert s.idle_seconds == 60
    assert s.scan_interval_seconds == 30


def test_strategy_override_accepts_single_idle_field() -> None:
    """One-sided override is allowed; the cross-field check is deferred
    to post-merge time (in apply_overrides) when both are known."""
    s = StrategyOverride(scan_interval_seconds=999)
    assert s.scan_interval_seconds == 999
    assert s.idle_seconds is None
