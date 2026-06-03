from __future__ import annotations

from everos.infra.ome.events import BaseEvent
from everos.infra.ome.exceptions import (
    EmitNotDeclaredError,
    OMEError,
    StartupValidationError,
)


class _UnknownEvent(BaseEvent):
    pass


def test_ome_error_is_base_exception() -> None:
    assert issubclass(OMEError, Exception)


def test_startup_validation_error_inherits_ome_error() -> None:
    assert issubclass(StartupValidationError, OMEError)


def test_emit_not_declared_error_inherits_ome_error() -> None:
    assert issubclass(EmitNotDeclaredError, OMEError)


def test_emit_not_declared_carries_strategy_and_event() -> None:
    ev = _UnknownEvent()
    err = EmitNotDeclaredError(strategy="cluster_memcells", event=ev)
    assert err.strategy == "cluster_memcells"
    assert err.event is ev
    assert "_UnknownEvent" in str(err)
    assert "cluster_memcells" in str(err)


def test_startup_validation_carries_message() -> None:
    err = StartupValidationError("missing trigger.on")
    assert "missing trigger.on" in str(err)
