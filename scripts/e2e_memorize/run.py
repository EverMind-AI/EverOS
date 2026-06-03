"""End-to-end memorize runner — in-process call into ``service.memorize``.

Calls ``service.memorize.memorize()`` directly (not via HTTP) so this works
without ``everos server start``. Drives a fixture through ``/add`` in
N-sized batches, then triggers ``/flush`` to drain the tail.

Reads ``settings.memorize.mode`` from current env / toml — set the mode via
``EVEROS_MEMORIZE__MODE=chat|agent`` *before* invoking this script (the
config is cached after the first ``load_settings()`` call).

Usage:
    EVEROS_MEMORIZE__MODE=chat  uv run python scripts/e2e_memorize/run.py \\
        --fixture scripts/e2e_memorize/fixtures/chat_session.json

    EVEROS_MEMORIZE__MODE=agent uv run python scripts/e2e_memorize/run.py \\
        --fixture scripts/e2e_memorize/fixtures/agent_session.json --batch-size 5

After it finishes, check:
    ~/.everos/users/<owner>/episodes/<date>.md      (written sync by 4A)
    ~/.everos/.index/sqlite/system.db memcell rows  (written by boundary)
    ~/.everos/agents/<agent>/agent_cases/<date>.md  (written async by OME
        - only if a consumer of AgentMemCellWritten is registered)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from sqlmodel import SQLModel

from everos.component.llm import get_llm_client
from everos.config import load_settings
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import dispose_engine, get_engine
from everos.service.memorize import _get_engine as _get_ome_engine
from everos.service.memorize import memorize


def _chunks(items: list[dict], n: int) -> list[list[dict]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _print_header(mode: str, fixture_path: Path, session_id: str) -> None:
    print("=" * 72)
    print(f"  everos e2e memorize  ·  mode={mode}")
    print(f"  fixture     : {fixture_path.name}")
    print(f"  session_id  : {session_id}")
    print(f"  memory root : {MemoryRoot.default().root}")
    llm_state = "<configured>" if get_llm_client() else "<None — pipeline will skip>"
    print(f"  llm_client  : {llm_state}")
    print("=" * 72)


def _list_written_files(session_id: str, mode: str) -> None:
    """Walk memory root and print files touched in this run."""
    root = MemoryRoot.default().root
    cutoff = time.time() - 600  # files modified in the last 10 min
    print()
    print("─── files modified within the last 10 minutes under memory root ───")
    interesting = ("users", "agents", "knowledge", ".index")
    for sub in interesting:
        base = Path(root) / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.stat().st_mtime >= cutoff:
                rel = p.relative_to(root)
                size = p.stat().st_size
                print(f"  {rel}  ({size}b)")
    print()
    print(f"Tip: grep '{session_id}' in any episode md to find this run's entries.")


async def _setup() -> None:
    """Create sqlite schema + start OME engine — the bits the HTTP lifespan
    normally handles. LanceDB is not needed for the memorize sync path
    (only cascade reads it), so we skip it.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    ome = _get_ome_engine()
    await ome.start()


async def _teardown() -> None:
    ome = _get_ome_engine()
    await ome.stop()
    await dispose_engine()


async def _run(args: argparse.Namespace) -> None:
    settings = load_settings()
    mode = settings.memorize.mode
    if args.expected_mode and args.expected_mode != mode:
        print(
            f"!! expected mode={args.expected_mode!r} but "
            f"settings.memorize.mode={mode!r}. "
            "Set EVEROS_MEMORIZE__MODE before launching."
        )
        sys.exit(2)

    fixture_path = Path(args.fixture).resolve()  # noqa: ASYNC240
    fixture = json.loads(fixture_path.read_text())  # noqa: ASYNC230
    messages: list[dict] = fixture["messages"]
    session_id = f"{fixture.get('session_id_hint', 'e2e')}_{uuid.uuid4().hex[:8]}"

    _print_header(mode, fixture_path, session_id)

    if args.dry_run:
        for i, batch in enumerate(_chunks(messages, args.batch_size), start=1):
            print(
                f"[dry] batch {i}: {len(batch)} msgs "
                f"(first content: {batch[0]['content'][:60]!r})"
            )
        print("[dry] would flush at the end")
        return

    await _setup()
    try:
        batches = _chunks(messages, args.batch_size)
        for i, batch in enumerate(batches, start=1):
            result = await memorize(
                {"session_id": session_id, "messages": batch}, is_final=False
            )
            print(
                f"add batch {i}/{len(batches)} ({len(batch)} msgs)  →  "
                f"status={result.status:<11s}  message_count={result.message_count}"
            )

        print()
        print("flushing residual tail...")
        flush_result = await memorize(
            {"session_id": session_id, "messages": []}, is_final=True
        )
        print(
            f"flush  →  status={flush_result.status:<11s}  "
            f"message_count={flush_result.message_count}"
        )

        # OME strategies are fire-and-forget; each cell fires 2 strategies
        # (atomic_facts + foresight), each ~5-10s on a real LLM. Sleep long
        # enough for ~8-10 invocations to finish before engine.stop() drains
        # the scheduler — otherwise APS cancels in-flight LLM calls.
        await asyncio.sleep(30)

        _list_written_files(session_id, mode)
    finally:
        await _teardown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        required=True,
        help="path to fixture JSON (e.g. fixtures/chat_session.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="how many messages per /add call (default 6 — 20 msgs across ~4 batches)",
    )
    parser.add_argument(
        "--expected-mode",
        choices=["chat", "agent"],
        help="sanity check: fail fast if settings.memorize.mode mismatches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the batch plan without calling memorize",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
