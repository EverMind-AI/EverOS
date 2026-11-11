"""The store-version gate: EverOS 2 opens only the stores EverOS 2 built.

``<root>/.index/store_version`` holds the store's major version, one integer
(``2``). Every process that opens a root -- the server, every ``everos
cascade`` command, backfill -- passes this gate before it touches anything:
before the system DB gains a table or a column, before an index table is
created or migrated, before the cascade reads a file.

* A marker that says ``2`` opens. Changes inside a major keep migrating as
  they always have (LanceDB ``.table_schema_version``, SQLite added columns).
* A marker that says anything else is refused: ``1`` (or a newer major) is
  another EverOS's store.
* No marker and no store state -- no system DB, no index table: a new store,
  stamped. Markdown already in the root (hand-written, or left after
  ``.index`` was wiped) is indexed by this build, as any markdown is: the
  gate judges a store by its state, and markdown carries none.
* No marker but store state: a store from before this gate. It is adopted
  (stamped) only when something in it can have been written by EverOS 2
  alone -- an index table schema of v3 or later, the ``image_memory`` /
  ``group_profile_item`` index tables, one of the system DB tables 2.0
  added, the ``figures/`` picture store. EverOS 1.x wrote none of these.
  Anything else is a 1.x store and is refused, unchanged: 2.0's extraction
  and prompts differ, so a 1.x store is rebuilt from its source
  conversations rather than upgraded in place (no silent LanceDB v2 -> v5
  migration).
"""

from __future__ import annotations

__all__ = ["STORE_VERSION", "StoreVersion"]

import contextlib
import os
import sqlite3
from pathlib import Path

from everos.core.errors import StoreVersionError
from everos.core.observability.logging import get_logger

from .memory_root import MemoryRoot

logger = get_logger(__name__)

STORE_VERSION = 2
"""The store major this build opens and writes."""

_MARKER = "store_version"

_TABLE_SCHEMA_MARKER = ".table_schema_version"
"""LanceDB's own marker; versions 3 and later exist only in EverOS 2."""

_FIRST_2_0_TABLE_SCHEMA = 3

_2_0_INDEX_TABLES = ("image_memory.lance", "group_profile_item.lance")
"""LanceDB tables EverOS 1.x never created."""

_2_0_SYSTEM_TABLES = frozenset(
    {
        "image_original",
        "forget_log",
        "message_forget",
        "group_meta",
        "group_member",
        "session_binding",
    }
)
"""System DB tables EverOS 1.x never created."""


class StoreVersion:
    """The gate over one memory root.

    Args:
        root: The memory root the process is about to open.
    """

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root

    @property
    def marker(self) -> Path:
        """``<root>/.index/store_version``."""
        return self._root.index_dir / _MARKER

    def check(self) -> None:
        """Let this root be opened, or refuse it before anything in it changes.

        Writes the marker for a new store and for an adopted one.

        Raises:
            StoreVersionError: The root holds another major's store, a 1.x
                store, or a marker that is not a version.
        """
        found = self._read()
        if found == STORE_VERSION:
            return
        if found is not None:
            raise StoreVersionError(self._other_major(found))
        if not self._has_store_state():
            self._write()
            logger.info("store_version_created", root=str(self._root.root))
            return
        evidence = self._evidence_of_2_0()
        if not evidence:
            raise StoreVersionError(self._not_2_0())
        self._write()
        logger.warning(
            "store_version_adopted",
            root=str(self._root.root),
            evidence=evidence,
            reason="a store from before the store-version gate, written by EverOS 2",
        )

    # ── reading the root ──────────────────────────────────────────────────

    def _read(self) -> int | None:
        try:
            text = self.marker.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StoreVersionError(
                f"{self.marker} cannot be read ({exc}); this EverOS opens a "
                "memory root only once it knows the store's version."
            ) from exc
        try:
            return int(text)
        except ValueError:
            raise StoreVersionError(
                f"{self.marker} holds {text[:40]!r}, not a store version. "
                f"A store written by this EverOS holds {STORE_VERSION}."
            ) from None

    def _has_store_state(self) -> bool:
        """Whether a build has opened this root: a system DB or an index table."""
        root = self._root
        if root.system_db.exists():
            return True
        return root.lancedb_dir.is_dir() and any(root.lancedb_dir.glob("*.lance"))

    def _evidence_of_2_0(self) -> list[str]:
        """What in this root only EverOS 2 can have written; empty for 1.x."""
        root = self._root
        evidence: list[str] = []
        schema = _table_schema_version(root.lancedb_dir / _TABLE_SCHEMA_MARKER)
        if schema is not None and schema >= _FIRST_2_0_TABLE_SCHEMA:
            evidence.append(f"index table schema v{schema}")
        evidence.extend(
            f"index table {name}"
            for name in _2_0_INDEX_TABLES
            if (root.lancedb_dir / name).is_dir()
        )
        evidence.extend(
            f"system table {name}"
            for name in sorted(_system_tables(root.system_db) & _2_0_SYSTEM_TABLES)
        )
        if (root.root / "figures").is_dir():
            evidence.append("figures/")
        return evidence

    # ── writing the marker ────────────────────────────────────────────────

    def _write(self) -> None:
        """Write the marker atomically; a failure is logged, never fatal.

        The store was accepted on its contents, so an unwritable marker only
        means the next start decides the same way again.
        """
        target = self.marker
        staging = target.with_name(f".{_MARKER}.{os.getpid()}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.write_text(f"{STORE_VERSION}\n", encoding="utf-8")
            os.replace(staging, target)
        except OSError as exc:
            with contextlib.suppress(OSError):
                staging.unlink()
            logger.warning("store_version_unwritten", path=str(target), error=str(exc))

    # ── messages ──────────────────────────────────────────────────────────

    def _other_major(self, found: int) -> str:
        newer = found > STORE_VERSION
        return (
            f"This memory root holds an EverOS {found} store "
            f"({self.marker} = {found}); this EverOS opens EverOS "
            f"{STORE_VERSION} stores only, and nothing in the root was changed. "
            + (
                "Run it with the newer EverOS that built it."
                if newer
                else "Keep serving it with the EverOS that built it, or rebuild "
                "its memory with this EverOS from the original conversations "
                "into a new root."
            )
        )

    def _not_2_0(self) -> str:
        schema = _table_schema_version(self._root.lancedb_dir / _TABLE_SCHEMA_MARKER)
        index = f"index table schema v{schema}" if schema is not None else "no index"
        return (
            f"This memory root was not built by EverOS {STORE_VERSION}: "
            f"{self._root.root}\n"
            f"It has no {self.marker.relative_to(self._root.root)} and nothing "
            f"in it that only EverOS {STORE_VERSION} writes ({index}, no "
            f"{STORE_VERSION}.0 table): it is an EverOS 1.x store. EverOS "
            f"{STORE_VERSION} opens only the stores it built -- its extraction "
            "and prompts changed, so a 1.x store is not upgraded in place -- "
            "and nothing in this root was changed. Keep serving it with EverOS "
            "1.x, or rebuild its memory with this EverOS from the original "
            "conversations into a new root."
        )


def _table_schema_version(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _system_tables(path: Path) -> set[str]:
    """The table names of a system DB, read without writing to it.

    ``immutable=1``: SQLite opens neither the WAL nor its shared-memory file,
    so a refused store is left exactly as it was -- its ``-shm`` file
    included. A table still only in the WAL is not seen; the index evidence
    covers such a store.
    """
    if not path.is_file():
        return set()
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro&immutable=1", uri=True, timeout=5
        )
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.debug("store_version_system_db_unread", path=str(path), error=str(exc))
        return set()
    return {str(name) for (name,) in rows}
