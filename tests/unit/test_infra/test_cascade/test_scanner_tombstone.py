"""A path present at decision time must not be tombstoned (#544)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from everos.infra.cascade import scanner as scanner_mod
from everos.infra.cascade.reconciler import PriorState
from everos.infra.cascade.scanner import CascadeScanner
from everos.infra.persistence import MemoryRoot


class _Ledger:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    async def upsert(
        self,
        md_path: str,
        *,
        kind: str,
        change_type: str,
        mtime: float,
    ) -> int:
        self.upserts.append(
            {
                "md_path": md_path,
                "kind": kind,
                "change_type": change_type,
                "mtime": mtime,
            }
        )
        return 1


def _write_file(root: MemoryRoot, rel: str) -> Path:
    path = root.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("content", encoding="utf-8")
    return path


async def test_present_file_is_rechecked_instead_of_tombstoned(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    rel = "users/u1/episodes/episode-2026-01-01.md"
    _write_file(root, rel)

    ledger = _Ledger()
    scanner = CascadeScanner(root, ledger)  # type: ignore[arg-type]

    async def _state(_: object) -> dict[str, PriorState]:
        return {
            rel: PriorState(
                md_path=rel,
                kind="episode",
                mtime=1.0,
                status="done",
                change_type="added",
            )
        }

    # The walk missed the file (rename gap); the state snapshot still has it.
    monkeypatch.setattr(scanner_mod, "_collect_scan_inputs", lambda *_: [])
    monkeypatch.setattr(scanner_mod, "_load_state_snapshot", _state)

    decisions = await scanner.scan_once()

    assert len(decisions) == 1
    assert decisions[0].change_type == "modified"
    assert decisions[0].mtime > 1.0
    assert ledger.upserts == [
        {
            "md_path": rel,
            "kind": "episode",
            "change_type": "modified",
            "mtime": decisions[0].mtime,
        }
    ]


async def test_actually_missing_file_stays_deleted(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    rel = "users/u1/episodes/gone.md"

    ledger = _Ledger()
    scanner = CascadeScanner(root, ledger)  # type: ignore[arg-type]

    async def _state(_: object) -> dict[str, PriorState]:
        return {
            rel: PriorState(
                md_path=rel,
                kind="episode",
                mtime=1.0,
                status="done",
                change_type="added",
            )
        }

    monkeypatch.setattr(scanner_mod, "_collect_scan_inputs", lambda *_: [])
    monkeypatch.setattr(scanner_mod, "_load_state_snapshot", _state)

    decisions = await scanner.scan_once()

    assert len(decisions) == 1
    assert decisions[0].change_type == "deleted"
