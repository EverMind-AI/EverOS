from __future__ import annotations

from datetime import datetime

from everos.infra.ome.events import BaseEvent, CronTick, IdleTick, ManualTick


class _DemoEvent(BaseEvent):
    payload: str


def test_base_event_auto_generates_id_and_ts() -> None:
    e = _DemoEvent(payload="x")
    assert isinstance(e.event_id, str)
    assert len(e.event_id) >= 32
    assert isinstance(e.ts, datetime)
    assert e.ts.tzinfo is not None


def test_base_event_is_frozen() -> None:
    e = _DemoEvent(payload="x")
    try:
        e.payload = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("BaseEvent should be frozen")


def test_base_event_round_trips_json() -> None:
    e = _DemoEvent(payload="hello")
    blob = e.model_dump_json()
    restored = _DemoEvent.model_validate_json(blob)
    assert restored == e


def test_cron_tick_carries_strategy_name() -> None:
    e = CronTick(strategy_name="profile_extraction")
    assert e.strategy_name == "profile_extraction"


def test_idle_tick_carries_bucket_and_seconds() -> None:
    e = IdleTick(strategy_name="agent_skill", bucket_key="user_42", idle_seconds=900)
    assert e.bucket_key == "user_42"
    assert e.idle_seconds == 900


def test_manual_tick_carries_strategy_name() -> None:
    e = ManualTick(strategy_name="cluster_memcells")
    assert e.strategy_name == "cluster_memcells"
