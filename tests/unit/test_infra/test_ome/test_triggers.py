from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.infra.ome.events import BaseEvent
from everos.infra.ome.triggers import Cron, Idle, Immediate


class _A(BaseEvent):
    pass


class _B(BaseEvent):
    pass


class _EventWithUserId(BaseEvent):
    user_id: str


def test_immediate_accepts_event_classes() -> None:
    t = Immediate(on=[_A, _B])
    assert t.on == [_A, _B]


def test_immediate_rejects_empty_on() -> None:
    with pytest.raises(ValidationError):
        Immediate(on=[])


def test_cron_accepts_expression() -> None:
    t = Cron(expr="0 3 * * *")
    assert t.expr == "0 3 * * *"


def test_cron_rejects_blank() -> None:
    with pytest.raises(ValidationError):
        Cron(expr="")


def test_idle_defaults_scan_interval() -> None:
    t = Idle(on=[_EventWithUserId], event_field="user_id", idle_seconds=900)
    assert t.scan_interval_seconds == 60


def test_idle_rejects_negative_idle_seconds() -> None:
    with pytest.raises(ValidationError):
        Idle(on=[_EventWithUserId], event_field="user_id", idle_seconds=-1)


def test_cron_accepts_valid_crontab() -> None:
    t = Cron(expr="0 3 * * *")
    assert t.expr == "0 3 * * *"


def test_cron_rejects_invalid_crontab() -> None:
    with pytest.raises(ValidationError) as exc:
        Cron(expr="not a cron")
    assert "expr" in str(exc.value)


def test_cron_rejects_out_of_range_field() -> None:
    # APS raises on out-of-range fields (e.g. minute=60)
    with pytest.raises(ValidationError):
        Cron(expr="60 0 * * *")


def test_idle_accepts_existing_event_field() -> None:
    t = Idle(
        on=[_EventWithUserId],
        event_field="user_id",
        idle_seconds=30,
        scan_interval_seconds=10,
    )
    assert t.event_field == "user_id"


def test_idle_rejects_missing_event_field() -> None:
    with pytest.raises(ValidationError) as exc:
        Idle(on=[_EventWithUserId], event_field="bad_name", idle_seconds=30)
    msg = str(exc.value)
    assert "bad_name" in msg
    assert "user_id" in msg


def test_idle_validator_runs_on_model_validate() -> None:
    base = Idle(
        on=[_EventWithUserId],
        event_field="user_id",
        idle_seconds=30,
        scan_interval_seconds=10,
    )
    with pytest.raises(ValidationError):
        Idle.model_validate({**base.model_dump(), "event_field": "nope"})


class _AnotherEventWithUserId(BaseEvent):
    user_id: str


class _EventWithoutUserId(BaseEvent):
    other: str


def test_idle_accepts_multiple_event_classes() -> None:
    t = Idle(
        on=[_EventWithUserId, _AnotherEventWithUserId],
        event_field="user_id",
        idle_seconds=30,
        scan_interval_seconds=10,
    )
    assert t.on == [_EventWithUserId, _AnotherEventWithUserId]


def test_idle_rejects_event_field_missing_in_any_class() -> None:
    with pytest.raises(ValidationError) as exc:
        Idle(
            on=[_EventWithUserId, _EventWithoutUserId],
            event_field="user_id",
            idle_seconds=30,
            scan_interval_seconds=10,
        )
    msg = str(exc.value)
    assert "user_id" in msg
    assert "_EventWithoutUserId" in msg


def test_idle_rejects_scan_interval_exceeding_half_idle() -> None:
    """The Idle docstring promises scan cadence <= idle_seconds // 2 so the
    scanner has at least two chances to observe an idle bucket before its
    silence window expires."""
    with pytest.raises(ValidationError, match="scan_interval_seconds"):
        Idle(
            on=[_EventWithUserId],
            event_field="user_id",
            idle_seconds=30,
            scan_interval_seconds=20,
        )


def test_idle_accepts_scan_interval_at_half_idle() -> None:
    """Boundary: scan_interval_seconds == idle_seconds // 2 is accepted."""
    t = Idle(
        on=[_EventWithUserId],
        event_field="user_id",
        idle_seconds=60,
        scan_interval_seconds=30,
    )
    assert t.scan_interval_seconds == 30
