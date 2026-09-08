"""The switch that lets a server start without the cascade projector.

On a retrieval-only server cascade's periodic scan can re-enqueue a whole store's
markdown, which starves search on a dense store. The switch is deliberately scoped to
cascade; OME keeps its normal process lifecycle.

Off by default, so an ingesting daemon is unaffected. That default is the part worth
pinning hardest: a switch that defaults to "on" would silently stop extraction.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

from everos.entrypoints.api.lifespans.cascade import CascadeLifespanProvider

TRUTHY = ["1", "true", "TRUE", "yes", "Yes"]
FALSY = ["", "   ", "0", "false", "no", "off", "maybe"]


@pytest.mark.parametrize("value", TRUTHY)
async def test_cascade_startup_is_skipped_when_disabled(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE", value)
    provider = CascadeLifespanProvider()
    assert await provider.startup(FastAPI()) is None
    assert provider._orchestrator is None


@pytest.mark.parametrize("value", FALSY)
async def test_an_unrecognised_value_does_not_disable_cascade(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything that is not clearly "yes" must leave ingestion working.

    Reaching startup proves the gate did not fire; what it does after that needs a real
    store, so the call is expected to get further and fail on that instead.
    """
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE", value)
    provider = CascadeLifespanProvider()
    reached: dict[str, Any] = {}

    def _boom() -> Any:
        reached["past_the_gate"] = True
        raise RuntimeError("stop here")

    monkeypatch.setattr(
        "everos.entrypoints.api.lifespans.cascade.MemoryRoot.resolve", _boom
    )
    with pytest.raises(RuntimeError, match="stop here"):
        await provider.startup(FastAPI())
    assert reached.get("past_the_gate") is True


async def test_unset_leaves_cascade_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default. A run that ingests must not have to know these exist."""
    monkeypatch.delenv("EVEROS_DISABLE_CASCADE", raising=False)

    cascade = CascadeLifespanProvider()
    monkeypatch.setattr(
        "everos.entrypoints.api.lifespans.cascade.MemoryRoot.resolve",
        lambda: (_ for _ in ()).throw(RuntimeError("cascade gate open")),
    )
    with pytest.raises(RuntimeError, match="cascade gate open"):
        await cascade.startup(FastAPI())
