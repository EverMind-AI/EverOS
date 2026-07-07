from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from everos.component.utils.datetime import get_now_with_timezone, get_utc_now
from everos.infra.ome.records import RunRecord, RunStatus, StrategyRouteInfo


def _ok_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build a baseline-valid RunRecord kwargs dict."""
    base: dict[str, Any] = {
        "run_id": "r1",
        "strategy_name": "s",
        "status": RunStatus.RUNNING,
        "attempt": 0,
        "started_at": get_now_with_timezone(),
        "event_topic": "x:Y",
        "event_payload": "{}",
        "max_retries_snapshot": 1,
        "event_id": "evt_test",
    }
    base.update(overrides)
    return base


def test_run_status_values() -> None:
    assert RunStatus.RUNNING.value == "running"
    assert RunStatus.SUCCESS.value == "success"
    assert RunStatus.FAILED.value == "failed"
    assert RunStatus.DEAD_LETTER.value == "dead_letter"
    assert RunStatus.CRASHED.value == "crashed"


def test_run_record_minimal() -> None:
    rec = RunRecord(
        run_id="r1",
        strategy_name="cluster",
        status=RunStatus.RUNNING,
        attempt=0,
        started_at=get_now_with_timezone(),
        event_topic="my_app.events:EpisodeSaved",
        event_payload="{}",
        max_retries_snapshot=1,
        event_id="evt_test",
    )
    assert rec.finished_at is None
    assert rec.error is None


def test_run_record_round_trips_json() -> None:
    rec = RunRecord(
        run_id="r1",
        strategy_name="cluster",
        status=RunStatus.SUCCESS,
        attempt=0,
        started_at=get_now_with_timezone(),
        finished_at=get_now_with_timezone(),
        event_topic="x:Y",
        event_payload='{"a":1}',
        max_retries_snapshot=1,
        event_id="evt_test",
    )
    blob = rec.model_dump_json()
    restored = RunRecord.model_validate_json(blob)
    assert restored == rec


def test_strategy_route_info() -> None:
    info = StrategyRouteInfo(
        strategy_name="profile_extraction",
        enabled_pass=True,
        applies_to_pass=True,
        counter_pass=False,
        counter_progress=(3, 5),
    )
    assert info.will_run is False
    assert info.counter_progress == (3, 5)


# ---------------------------------------------------------------------------
# Constraint enforcement: every Field(...) / validator must actually reject
# the bad input it claims to reject. Add a test per declared constraint so
# accidental relaxation in the future fails CI.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["run_id", "strategy_name", "event_topic", "event_payload"],
)
def test_run_record_rejects_empty_identifier(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        RunRecord(**_ok_kwargs(**{field: ""}))


@pytest.mark.parametrize("field", ["attempt", "max_retries_snapshot"])
def test_run_record_rejects_negative_int(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        RunRecord(**_ok_kwargs(**{field: -1}))


def test_run_record_rejects_naive_started_at() -> None:
    naive = datetime(2026, 5, 12, 12, 0, 0)
    with pytest.raises(ValidationError, match="started_at"):
        RunRecord(**_ok_kwargs(started_at=naive))


def test_run_record_rejects_empty_error_when_set() -> None:
    """error=None is allowed; error="" is not (min_length=1)."""
    with pytest.raises(ValidationError, match="error"):
        RunRecord(
            **_ok_kwargs(
                status=RunStatus.FAILED,
                finished_at=get_now_with_timezone(),
                error="",
            )
        )


# Status-invariant violations: each must be rejected by _check_status_invariants.


def test_running_must_have_no_finished_at() -> None:
    with pytest.raises(ValidationError, match=r"RUNNING.*finished_at"):
        RunRecord(
            **_ok_kwargs(status=RunStatus.RUNNING, finished_at=get_now_with_timezone())
        )


def test_running_must_have_no_error() -> None:
    with pytest.raises(ValidationError, match=r"RUNNING.*error"):
        RunRecord(**_ok_kwargs(status=RunStatus.RUNNING, error="boom"))


def test_success_must_have_finished_at() -> None:
    with pytest.raises(ValidationError, match=r"success.*finished_at"):
        RunRecord(**_ok_kwargs(status=RunStatus.SUCCESS))


def test_success_must_have_no_error() -> None:
    with pytest.raises(ValidationError, match=r"SUCCESS.*error"):
        RunRecord(
            **_ok_kwargs(
                status=RunStatus.SUCCESS,
                finished_at=get_now_with_timezone(),
                error="should not be here",
            )
        )


@pytest.mark.parametrize(
    "status",
    [RunStatus.FAILED, RunStatus.DEAD_LETTER, RunStatus.CRASHED],
)
def test_terminal_failure_must_have_finished_at(status: RunStatus) -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        RunRecord(**_ok_kwargs(status=status, error="boom"))


@pytest.mark.parametrize(
    "status",
    [RunStatus.FAILED, RunStatus.DEAD_LETTER, RunStatus.CRASHED],
)
def test_terminal_failure_must_have_error(status: RunStatus) -> None:
    with pytest.raises(ValidationError, match="error"):
        RunRecord(**_ok_kwargs(status=status, finished_at=get_now_with_timezone()))


def test_strategy_route_info_rejects_empty_strategy_name() -> None:
    with pytest.raises(ValidationError, match="strategy_name"):
        StrategyRouteInfo(
            strategy_name="",
            enabled_pass=True,
            applies_to_pass=True,
            counter_pass=True,
        )


def test_run_record_accepts_event_id() -> None:
    rec = RunRecord(
        run_id="r1",
        strategy_name="s",
        status=RunStatus.RUNNING,
        attempt=0,
        started_at=get_utc_now(),
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
        event_id="abc123",
    )
    assert rec.event_id == "abc123"


def test_run_record_accepts_empty_event_id_for_migration_compat() -> None:
    """Empty event_id is valid for pre-existing rows migrated from older schema."""
    rec = RunRecord(**_ok_kwargs(event_id=""))
    assert rec.event_id == ""
