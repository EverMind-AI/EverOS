"""The two switches that let a server run read-only.

Both subsystems are write-side, and on a retrieval-only server both actively hurt rather
than merely idle: cascade's periodic scan re-enqueues a whole store's markdown, which
starved search on a dense store badly enough to hold it at zero; the OME engine holds an
exclusive per-store lock, which stops a second server from sharing one pre-built store
root -- the shape a parallel-lane evaluation needs.

Off by default, so an ingesting daemon is unaffected. That default is the part worth
pinning hardest: a switch that defaults to "on" would silently stop extraction.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

from everos.entrypoints.api.lifespans.cascade import CascadeLifespanProvider
from everos.entrypoints.api.lifespans.ome import OmeLifespanProvider

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


@pytest.mark.parametrize("value", TRUTHY)
async def test_ome_startup_is_skipped_when_disabled(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_DISABLE_OME", value)
    provider = OmeLifespanProvider()
    assert await provider.startup(FastAPI()) is None


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


@pytest.mark.parametrize("value", FALSY)
async def test_an_unrecognised_value_does_not_disable_ome(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_DISABLE_OME", value)
    provider = OmeLifespanProvider()
    reached: dict[str, Any] = {}

    def _boom(*_a: Any, **_k: Any) -> Any:
        reached["past_the_gate"] = True
        raise RuntimeError("stop here")

    monkeypatch.setattr("importlib.import_module", _boom)
    with pytest.raises(RuntimeError, match="stop here"):
        await provider.startup(FastAPI())
    assert reached.get("past_the_gate") is True


async def test_unset_leaves_both_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default. A run that ingests must not have to know these exist."""
    monkeypatch.delenv("EVEROS_DISABLE_CASCADE", raising=False)
    monkeypatch.delenv("EVEROS_DISABLE_OME", raising=False)

    cascade = CascadeLifespanProvider()
    monkeypatch.setattr(
        "everos.entrypoints.api.lifespans.cascade.MemoryRoot.resolve",
        lambda: (_ for _ in ()).throw(RuntimeError("cascade gate open")),
    )
    with pytest.raises(RuntimeError, match="cascade gate open"):
        await cascade.startup(FastAPI())

    # And OME, which the first version of this test named but never constructed.
    ome = OmeLifespanProvider()
    monkeypatch.setattr(
        "importlib.import_module",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ome gate open")),
    )
    with pytest.raises(RuntimeError, match="ome gate open"):
        await ome.startup(FastAPI())
