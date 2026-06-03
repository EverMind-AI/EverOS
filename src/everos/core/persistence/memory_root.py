"""memory-root path manager.

Single root directory holding all persisted memory:

    User-visible (no dot prefix, edited by humans / agents):
        agents/      per-agent records
        users/       per-user records
        knowledge/   global shared knowledge

    System-managed (dotfile prefix, hidden by default in ls / Finder):
        .index/             derived indexes (rebuildable from markdown)
            sqlite/         system.db (+ WAL/SHM), ome.db, ome.aps.db
            lancedb/        LanceDB tables
        .tmp/               atomic-write staging directory
        .lock               single-process lock anchor (created on demand by
                            ``memory_root_lock``)

    User-editable (at the root):
        ome.toml            OME strategy overrides (hot-reloaded)

The cascade queue, LSN watermark, and change audit all live in
``system.db`` (table ``md_change_state``), not in separate dotfiles.

The default location and tunables come from :class:`everos.config.Settings`
(loaded from ``config/default.toml`` + ``EVEROS_*`` environment variables);
:meth:`MemoryRoot.default` resolves the configured path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ── app / project directory-name convention ──────────────────────────────────
#
# A memory root is partitioned by ``<app>/<project>`` *before* the user-visible
# scope dirs (``agents`` / ``users`` / ``knowledge``), so memory for different
# (app, project) pairs never shares a directory. The reserved id ``"default"``
# materialises as ``default_app`` / ``default_project`` on disk (rather than a
# bare ``default``) so a default space is visually distinct from a user-named
# directory; every other id maps to itself.
#
# The mapping is symmetric: the cascade path parser reverses it (see
# :func:`app_id_from_dir`) to recover the ids from an on-disk path. The write
# side (here) and the read side (cascade) MUST stay in lockstep, or rebuilt
# rows carry app/project that disagree with what was written. ``default_app`` /
# ``default_project`` are therefore reserved directory names.
_DEFAULT_SCOPE_ID = "default"
_DEFAULT_APP_DIR = "default_app"
_DEFAULT_PROJECT_DIR = "default_project"

# Path to the shipped OME override template; copied to ``<root>/ome.toml`` on
# first ``ensure()`` so users have a real file to edit instead of having to
# create one from scratch. ``parents[2]`` is the ``src/everos/`` package root
# (memory_root.py sits at ``core/persistence/memory_root.py``).
_OME_TEMPLATE_PATH = Path(__file__).parents[2] / "config" / "default_ome.toml"


def app_dir_name(app_id: str) -> str:
    """Map an ``app_id`` to its on-disk directory name."""
    return _DEFAULT_APP_DIR if app_id == _DEFAULT_SCOPE_ID else app_id


def project_dir_name(project_id: str) -> str:
    """Map a ``project_id`` to its on-disk directory name."""
    return _DEFAULT_PROJECT_DIR if project_id == _DEFAULT_SCOPE_ID else project_id


def app_id_from_dir(dir_name: str) -> str:
    """Inverse of :func:`app_dir_name` — recover the ``app_id`` from a dir name."""
    return _DEFAULT_SCOPE_ID if dir_name == _DEFAULT_APP_DIR else dir_name


def project_id_from_dir(dir_name: str) -> str:
    """Inverse of :func:`project_dir_name` — recover the ``project_id``."""
    return _DEFAULT_SCOPE_ID if dir_name == _DEFAULT_PROJECT_DIR else dir_name


@dataclass(frozen=True, init=False)
class MemoryRoot:
    """Path manager for a memory-root directory.

    Constructor accepts any path-like (``str`` or ``Path``); it is normalised
    to an absolute, resolved ``Path`` so equality and hashing are stable
    regardless of how the caller spells the path. ``init=False`` is paired
    with a hand-written ``__init__`` so the input type (``Path | str``) is
    decoupled from the stored field type (``Path``) — stdlib dataclass has
    no converter slot, and Pyright would otherwise reject ``MemoryRoot(s)``
    where ``s`` is a ``str``.
    """

    root: Path

    def __init__(self, root: Path | str) -> None:
        # ``frozen=True`` forbids attribute assignment, so go through
        # ``object.__setattr__`` to install the normalised Path field.
        resolved = Path(root).expanduser().resolve()
        object.__setattr__(self, "root", resolved)

    @classmethod
    def default(cls) -> MemoryRoot:
        """Return the memory-root from :class:`everos.config.Settings`.

        The effective default lives in ``config/default.toml`` (``[memory]
        root``); environment variable ``EVEROS_MEMORY__ROOT`` overrides it.
        """
        # Lazy import to keep this module dependency-free at import time.
        from everos.config import load_settings

        return cls(load_settings().memory.root)

    # ── User-visible (partitioned by app / project) ──────────────────────────
    #
    # These take ``(app_id, project_id)`` because the scope dirs hang off the
    # ``<root>/<app>/<project>/`` prefix; they are request-level inputs, never
    # instance state. Both default to ``"default"`` so call sites that don't
    # yet carry scope still resolve to the default space.

    def agents_dir(self, app_id: str = "default", project_id: str = "default") -> Path:
        """``<root>/<app>/<project>/agents/`` — per-agent records."""
        return (
            self.root / app_dir_name(app_id) / project_dir_name(project_id) / "agents"
        )

    def users_dir(self, app_id: str = "default", project_id: str = "default") -> Path:
        """``<root>/<app>/<project>/users/`` — per-user records."""
        return self.root / app_dir_name(app_id) / project_dir_name(project_id) / "users"

    def knowledge_dir(
        self, app_id: str = "default", project_id: str = "default"
    ) -> Path:
        """``<root>/<app>/<project>/knowledge/`` — shared knowledge."""
        return (
            self.root
            / app_dir_name(app_id)
            / project_dir_name(project_id)
            / "knowledge"
        )

    # ── System-managed (dotfiles) ───────────────────────────────────────────

    @property
    def index_dir(self) -> Path:
        """``<root>/.index/`` — derived index root."""
        return self.root / ".index"

    @property
    def lancedb_dir(self) -> Path:
        """``<root>/.index/lancedb/`` — LanceDB table root."""
        return self.index_dir / "lancedb"

    @property
    def sqlite_dir(self) -> Path:
        """``<root>/.index/sqlite/`` — SQLite system DB root.

        Holds ``system.db`` plus its sidecars (``-wal`` / ``-shm`` in WAL
        mode). Symmetric with :attr:`lancedb_dir`.
        """
        return self.index_dir / "sqlite"

    @property
    def system_db(self) -> Path:
        """``<root>/.index/sqlite/system.db`` — SQLite DB for system
        state, audit log, task queue, LSN watermark, and other metadata.
        """
        return self.sqlite_dir / "system.db"

    @property
    def ome_db(self) -> Path:
        """``<root>/.index/sqlite/ome.db`` — SQLite DB backing the Offline
        Memory Engine's own state: run records, counter store, idle store.
        Symmetric with :attr:`system_db`.
        """
        return self.sqlite_dir / "ome.db"

    @property
    def ome_aps_db(self) -> Path:
        """``<root>/.index/sqlite/ome.aps.db`` — SQLite DB holding the
        APScheduler jobstore for the Offline Memory Engine. Split from
        :attr:`ome_db` so APS's sync SQLAlchemy writer and OME's async
        aiosqlite writer never contend for the same sqlite file lock.
        """
        return self.sqlite_dir / "ome.aps.db"

    @property
    def ome_config(self) -> Path:
        """``<root>/ome.toml`` — user-editable OME strategy overrides.

        Drop a file here to toggle strategies on/off or tweak per-strategy
        knobs (max_retries, gate, cron …) without restarting the server.
        The engine watches this file and hot-reloads changes within ~2 s.

        Example to disable foresight and user-profile extraction::

            [strategies.extract_foresight]
            enabled = false

            [strategies.extract_user_profile]
            enabled = false
        """
        return self.root / "ome.toml"

    @property
    def lock_file(self) -> Path:
        """``<root>/.lock`` — single-process exclusive lock anchor."""
        return self.root / ".lock"

    @property
    def tmp_dir(self) -> Path:
        """``<root>/.tmp/`` — staging directory for batch / multi-step writes.

        Note:
            ``MarkdownWriter`` does *not* use this for atomic single-file
            writes; it uses a same-directory temp file to guarantee a
            same-filesystem rename. This directory is reserved for callers
            that need scratch space outside any single target directory.
        """
        return self.root / ".tmp"

    # ── Operations ──────────────────────────────────────────────────────────

    def ensure(self) -> None:
        """Create the memory-root and the runtime-required dotfile dirs.

        User-visible directories (``agents/`` / ``users/`` / ``knowledge/``)
        are *not* pre-created — they appear on first write of their records.
        Only directories the runtime infrastructure requires are made:

            <root>/
            <root>/.index/
            <root>/.index/sqlite/
            <root>/.index/lancedb/
            <root>/.tmp/
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # Materialize the OME override template on first run; existence-only
        # check preserves any edits the user has already made.
        if not self.ome_config.exists():
            self.ome_config.write_bytes(_OME_TEMPLATE_PATH.read_bytes())
