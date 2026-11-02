"""Tests for :class:`everos.infra.cascade.DeleteFence`."""

from __future__ import annotations

from dataclasses import dataclass

import structlog.testing
from prometheus_client import REGISTRY

from everos.infra.cascade import DeleteFence
from everos.memory.ports import all_of, any_of, eq, one_of


@dataclass(frozen=True)
class _Row:
    app_id: str
    project_id: str
    id: str


def _gauge(name: str) -> float | None:
    return REGISTRY.get_sample_value(name)


def test_add_without_registered_run_keeps_nothing() -> None:
    """A19: with no run in flight a deletion leaves the fence empty."""
    fence = DeleteFence()
    fence.add([("a", "p", "u_1")])
    assert not fence.blocks(_Row("a", "p", "u_1"))
    assert _gauge("everos_memory_delete_fence_entries") == 0


def test_add_blocks_rows_of_runs_in_flight() -> None:
    fence = DeleteFence()
    run = fence.register_run()
    fence.add([("a", "p", "u_1")])
    assert fence.blocks(_Row("a", "p", "u_1"))
    assert not fence.blocks(_Row("a", "p2", "u_1"))  # another project's row
    assert _gauge("everos_memory_delete_fence_entries") == 1
    assert _gauge("everos_memory_delete_fence_runs") == 1
    fence.end_run(run)
    assert _gauge("everos_memory_delete_fence_runs") == 0


def test_entry_dropped_once_all_earlier_runs_end() -> None:
    """Spec 1 §5.4 item 5: an entry lives while a run older than it runs."""
    fence = DeleteFence()
    early = fence.register_run()
    fence.add([("a", "p", "u_1")])
    late = fence.register_run()  # started after the deletion: reads new md
    fence.end_run(early)
    assert not fence.blocks(_Row("a", "p", "u_1"))
    assert _gauge("everos_memory_delete_fence_entries") == 0
    fence.end_run(late)


def test_entry_kept_while_an_earlier_run_remains() -> None:
    fence = DeleteFence()
    early = fence.register_run()
    mid = fence.register_run()
    fence.add([("a", "p", "u_1")])
    fence.end_run(mid)
    assert fence.blocks(_Row("a", "p", "u_1"))
    fence.end_run(early)
    assert not fence.blocks(_Row("a", "p", "u_1"))


def test_failed_run_end_also_prunes() -> None:
    """TLA+ Fence_Fixed, first round: a run that ended by timing out must
    still release its entries (FenceBounded)."""
    fence = DeleteFence()
    run = fence.register_run()
    fence.add([("a", "p", "u_1")])
    try:
        raise TimeoutError
    except TimeoutError:
        fence.end_run(run)
    assert _gauge("everos_memory_delete_fence_entries") == 0


def test_filter_for_selects_exactly_the_rows() -> None:
    fence = DeleteFence()
    where = fence.filter_for(
        [_Row("a", "p", "u_1"), _Row("b", "q", "u_2"), _Row("a", "p", "u_3")]
    )
    assert where == any_of(
        all_of(eq("app_id", "a"), eq("project_id", "p"), one_of("id", ["u_1", "u_3"])),
        all_of(eq("app_id", "b"), eq("project_id", "q"), one_of("id", ["u_2"])),
    )


def test_warns_when_fence_exceeds_threshold() -> None:
    """A24 (c): above the configured threshold the fence logs once."""
    fence = DeleteFence(warn_entries=3)
    run = fence.register_run()
    with structlog.testing.capture_logs() as logs:
        fence.add([("a", "p", f"u_{i}") for i in range(3)])
        assert not [e for e in logs if e["event"] == "memory_delete_fence_large"]
        fence.add([("a", "p", "u_3")])
        fence.add([("a", "p", "u_4")])
    warned = [e for e in logs if e["event"] == "memory_delete_fence_large"]
    assert [(e["entries"], e["threshold"]) for e in warned] == [(4, 3)]
    fence.end_run(run)
