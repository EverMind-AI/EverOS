"""Construction snapshots never affect business execution and isolate requests."""

from __future__ import annotations

import asyncio

from everos.memory.extract.trace import collect_construction_trace, record_construction


def test_disabled_trace_skips_snapshot_serialization() -> None:
    def fail() -> dict:
        raise AssertionError("Serialization must not run without a collector.")

    record_construction("boundary", snapshot_factory=fail)


def test_snapshot_failure_is_isolated_and_results_are_immutable() -> None:
    def fail() -> dict:
        raise ValueError("bad snapshot")

    value = {"content": ["Friday"]}
    with collect_construction_trace() as events:
        record_construction("boundary", snapshot_factory=fail)
        record_construction("extraction_result", result=value)
        value["content"].append("Monday")
    assert len(events) == 1
    assert events[0]["result"]["content"] == ["Friday"]


async def test_concurrent_construction_collectors_do_not_mix() -> None:
    async def run(owner: str) -> list[dict]:
        with collect_construction_trace() as events:
            await asyncio.sleep(0)
            record_construction("extraction_result", owner=owner, content="")
        return events

    alice, bob = await asyncio.gather(run("alice"), run("bob"))
    assert [event["owner"] for event in alice] == ["alice"]
    assert [event["owner"] for event in bob] == ["bob"]
