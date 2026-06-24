"""RunRecord / RunStatus / StrategyRouteInfo / CounterProgress — pure data classes.

Persistence in _stores/run_record.py.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, NamedTuple, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)


class RunStatus(StrEnum):
    """Terminal-or-running state of a single strategy run."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CRASHED = "crashed"


class RunRecord(BaseModel):
    """One row of the run_record table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: Annotated[str, Field(min_length=1)]
    strategy_name: Annotated[str, Field(min_length=1)]
    status: RunStatus
    attempt: Annotated[int, Field(ge=0)]
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    error: Annotated[str, Field(min_length=1)] | None = None
    event_topic: Annotated[
        str,
        Field(
            min_length=1,
            description="Stable cross-process event identifier in "
            "``<module>:<class>`` form (see ``BaseEvent.topic()``).",
        ),
    ]
    event_payload: Annotated[
        str,
        Field(
            min_length=1,
            description="JSON-encoded event (``BaseEvent.model_dump_json`` output).",
        ),
    ]
    max_retries_snapshot: Annotated[int, Field(ge=0)]
    event_id: str = ""

    @model_validator(mode="after")
    def _check_status_invariants(self) -> Self:
        if self.status == RunStatus.RUNNING:
            if self.finished_at is not None:
                raise ValueError("RunRecord: RUNNING must have finished_at=None")
            if self.error is not None:
                raise ValueError("RunRecord: RUNNING must have error=None")
        else:
            if self.finished_at is None:
                raise ValueError(f"RunRecord: {self.status} must have finished_at set")
            if self.status == RunStatus.SUCCESS:
                if self.error is not None:
                    raise ValueError("RunRecord: SUCCESS must have error=None")
            elif self.error is None:
                raise ValueError(f"RunRecord: {self.status} must have error set")
        return self


class CounterProgress(NamedTuple):
    """Per-bucket counter progress at inspect_dispatch time."""

    current: int
    threshold: int


class StrategyRouteInfo(BaseModel):
    """Per-strategy dispatch decision — returned by inspect_dispatch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_name: Annotated[str, Field(min_length=1)]
    enabled_pass: bool
    applies_to_pass: bool
    counter_pass: bool
    counter_progress: CounterProgress | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def will_run(self) -> bool:
        return self.enabled_pass and self.applies_to_pass and self.counter_pass
