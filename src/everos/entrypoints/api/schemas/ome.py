"""Request and response models for ``POST /ome/trigger``."""

from __future__ import annotations

from pydantic import BaseModel


class TriggerRequest(BaseModel):
    """Request body for ``POST /api/v2/ome/trigger``."""

    name: str
    timeout: float = 120.0
    force: bool = False


class RunSummary(BaseModel):
    """One strategy run within a trigger response."""

    run_id: str
    status: str
    """One of: running / success / failed / dead_letter / crashed."""
    error: str | None = None


class TriggerResponse(BaseModel):
    """Response body for ``POST /api/v2/ome/trigger``.

    ``status`` distinguishes three outcomes that were previously masked as
    a single ``ok``:

    - ``ok``: at least one strategy was dispatched and all runs settled
      within ``timeout``. Individual run outcomes are in ``runs`` (a
      ``dead_letter`` there is still ``ok`` at this level — the strategy
      *ran*, it just failed permanently).
    - ``timeout``: at least one strategy was dispatched but the engine did
      not go idle within ``timeout``. Runs may be partially complete;
      poll ``GET /health`` for cascade convergence separately.
    - ``not_dispatched``: no strategy was dispatched — the subscriber was
      rejected by one of the dispatch gates (``_routes_to`` / ``enabled`` /
      ``applies_to`` / ``Counter``). Common cause: forgetting
      ``force=true`` on a strategy that is ``enabled=false`` in ome.toml.
    """

    status: str
    """One of: ok / timeout / not_dispatched."""
    name: str
    dispatched: int
    """Number of strategy routes that were enqueued. ``0`` iff status is
    ``not_dispatched``."""
    runs: list[RunSummary] = []
    """One entry per strategy run *attempt*, not per dispatched route: a
    strategy that retried before settling contributes multiple entries
    sharing one ``event_id``. Includes dead-lettered runs whose errors
    would otherwise be invisible to the caller (they live in the SQLite
    ``run_record`` table with no HTTP surface until this field was
    added)."""
