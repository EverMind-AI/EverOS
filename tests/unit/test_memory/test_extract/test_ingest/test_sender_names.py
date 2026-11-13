"""Ingest names each message's sender the way the sender last named themselves.

Pins the rules of :mod:`everos.memory.extract.ingest.sender_names` through
``ingest.process``: what is recorded per ``(app, project, sender)``, which
messages are named from it, and that a sender never named stays ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence

from everos.memory.extract.ingest import process
from tests.fakes import InMemorySenderNameStore, unused_raw_file_builder

_TS = 1_775_116_800_000


async def _names(
    store: InMemorySenderNameStore,
    messages: Sequence[tuple[str, str | None]],
    *,
    app_id: str = "a",
) -> list[str | None]:
    """Ingest ``(sender_id, sender_name)`` messages; the names they come out with."""
    payload = {
        "session_id": "s1",
        "app_id": app_id,
        "project_id": "p",
        "messages": [
            {
                "sender_id": sender_id,
                "sender_name": name,
                "role": "user",
                "timestamp": _TS + i,
                "content": f"line {i}",
            }
            for i, (sender_id, name) in enumerate(messages)
        ],
    }
    ingested = await process(
        payload,
        multimodal_llm=None,
        raw_file_builder=unused_raw_file_builder,
        sender_names=store,
    )
    return [m.sender_name for m in ingested.messages]


async def test_a_name_in_the_batch_names_its_sender_lines_before_and_after() -> None:
    store = InMemorySenderNameStore()
    names = await _names(store, [("u1", None), ("u1", "Alice"), ("u1", None)])
    assert names == ["Alice", "Alice", "Alice"]
    assert store.names == {("a", "p", "u1"): "Alice"}


async def test_a_later_batch_without_names_takes_the_recorded_one() -> None:
    store = InMemorySenderNameStore()
    await _names(store, [("u1", "Alice")])
    assert await _names(store, [("u1", None), ("u1", "")]) == ["Alice", "Alice"]


async def test_a_sender_never_named_stays_unnamed() -> None:
    store = InMemorySenderNameStore()
    assert await _names(store, [("u1", None)]) == [None]
    assert store.names == {}


async def test_the_latest_name_wins_and_a_line_keeps_its_own() -> None:
    store = InMemorySenderNameStore()
    await _names(store, [("u1", "Ali")])
    names = await _names(store, [("u1", "Ally"), ("u1", None), ("u1", "Alice")])
    assert names == ["Ally", "Alice", "Alice"]
    assert store.names == {("a", "p", "u1"): "Alice"}


async def test_blank_names_are_absent_and_kept_names_are_stripped() -> None:
    store = InMemorySenderNameStore()
    await _names(store, [("u1", "Alice")])
    assert await _names(store, [("u1", "   "), ("u2", "  Bob ")]) == ["Alice", "Bob"]
    assert store.names == {("a", "p", "u1"): "Alice", ("a", "p", "u2"): "Bob"}


async def test_names_do_not_cross_spaces_or_senders() -> None:
    store = InMemorySenderNameStore()
    await _names(store, [("u1", "Alice")], app_id="a")
    assert await _names(store, [("u1", None)], app_id="other") == [None]
    assert await _names(store, [("u2", None)], app_id="a") == [None]
