---
name: add-memory-kind
description: Add a new business memory kind end-to-end. Pick the storage combination (Markdown / SQLite / LanceDB), pick the markdown strategy (daily-log / single-file), then wire up the schema(s), repo(s), and writer(s).
---

# /add-memory-kind — Add a new business memory kind

## When to invoke

Adding a new persisted business entity (Episode, ImageMemory, Scene,
Profile, a group-profile item, or something custom). Multiple storage
layers may be involved; this skill walks the decision then the wiring.

## 1. Decide the storage combination

A memory kind **does not have to use all three** layers. Pick by what
the kind actually needs:

| Need | Markdown | SQLite | LanceDB |
|---|:-:|:-:|:-:|
| Human-readable / agent-editable source-of-truth text | ✅ | | |
| Structured state, ACID transactions, joins, predicates | | ✅ | |
| Vector / BM25 / hybrid retrieval | | | ✅ |

Common combinations seen in EverOS:

| Combo | Example | Rationale |
|---|---|---|
| **md only** | a group profile's `pending.md` (the pending batch) | text-of-truth, no index needed |
| **md + lancedb** | episode / image_memory / profile (`user.md` → `user_profile`) / group-profile item | text-of-truth + retrieval |
| **md + sqlite** | group roster (`group.md` → `group_meta` / `group_member` / `session_binding`) | text-of-truth + structured state to query |
| **md + sqlite + lancedb** | scene (versions in markdown, rows in the `episode` table, clusters in SQLite) | when you need *both* transactional state AND retrieval |
| **sqlite only** | memcell ledger / memorize queue / cascade LSN watermark | system state, never user-facing |
| **lancedb only** | rare; usually you still want md | derived embeddings without text-of-truth |

Rule of thumb: **markdown is the truth**; sqlite and lancedb are derived
indexes that can be rebuilt from md. Drop md only when the kind has no
human-readable form (pure system state).

## 2. Pick the markdown storage strategy (if md is in your combo)

Two strategies — declared in the EverOS Markdown First spec:

| Strategy | Filename | Mutation | Examples |
|---|---|---|---|
| **Daily-log append** | `<prefix>-YYYY-MM-DD.md` | append entries | episode / image_memory / scene / group_profile, user and group tracks |
| **Single-file rewrite** | `user.md` / `groups/<g>/members/<m>.md` | overwrite the file | a person's profile in a space |

This skill has a **complete recipe for daily-log append**. The
single-file strategy is sketched in section 6.

---

## 3. Markdown daily-log: 4 steps

### 3.1 Frontmatter schema — `infra/persistence/markdown/_mds/<name>.py`

The episode kind is the model; the snippets below are its real schema and
writer, abridged.

```python
"""Episode daily-log frontmatter."""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from everos.infra.persistence.markdown._base.frontmatter import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class EpisodeDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """``users/<u>/episodes/episode-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "ep"
    DIR_NAME: ClassVar[str] = "episodes"
    FILE_PREFIX: ClassVar[str] = "episode"

    type: Literal["episode_daily"] = "episode_daily"
    date: _dt.date
    entry_count: int = 0
    last_appended_at: _dt.datetime | None = None
```

`DailyLogPathMixin` goes first, so `path_glob()` (what the cascade scans)
resolves to it. For a group-owned kind subclass `GroupScopedFrontmatter`
instead (`groups/<g>/...`). When the user and group tracks share a kind
name, give each a **distinct** `ENTRY_ID_PREFIX` (`ep` vs `gep`, `im` vs
`gim`, `sc` vs `gsc`): a row id is `<owner>_<entry_id>`, and users and
groups share one id space.

### 3.2 Re-export — `_mds/__init__.py`

```python
from .episode import EpisodeDailyFrontmatter as EpisodeDailyFrontmatter
```

### 3.3 Business writer — `infra/persistence/markdown/_writers/<name>.py`

```python
"""Episode daily-log writer."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from typing import Any

from everos.core.utils.datetime import get_now_with_timezone, to_iso_format

from .._mds import EpisodeDailyFrontmatter
from .base import BaseDailyWriter


class EpisodeWriter(BaseDailyWriter):
    schema = EpisodeDailyFrontmatter

    # The file-level frontmatter each append writes.
    def _frontmatter_updates(
        self, scope_id: str, date: _dt.date, *, next_count: int
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"episode_log_{scope_id}_{date.isoformat()}",
            "type": "episode_daily",
            "user_id": scope_id,
            "track": "user",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }
```

The entry count defaults to the entry markers already in the file
(`BaseDailyWriter._current_count`, async); override it only to trust a
frontmatter field instead.

### 3.4 Re-export — `_writers/__init__.py`

```python
from .episode_writer import EpisodeWriter as EpisodeWriter
```

### Done — usage

```python
from everos.infra.persistence.markdown._writers import EpisodeWriter

writer = EpisodeWriter(memory_root)
eid = await writer.append("u_jason", "I went to the doctor today.")
# → default_app/default_project/users/u_jason/episodes/episode-<today>.md
# → entry markers carry an auto-generated EntryId (e.g. ep_20260507_00000001)
```

---

## 4. (Optional) SQLite table — 4 steps

Skip this section if the kind doesn't need structured state beyond markdown.

### 4.1 Schema — `infra/persistence/sqlite/_tables/<name>.py`

```python
from everos.infra.persistence.sqlite import BaseTable, Field


class EpisodeState(BaseTable, table=True):
    __tablename__ = "episode_state"  # type: ignore[assignment]

    id: int | None = Field(default=None, primary_key=True)
    entry_id: str = Field(index=True, unique=True)
    cluster_id: str | None = Field(default=None, index=True)
    status: str = Field(default="active")
```

`BaseTable` already provides `created_at` / `updated_at` (auto-bumped).

### 4.2 Re-export — `tables/__init__.py`

```python
from .episode import EpisodeState as EpisodeState
```

### 4.3 Repo — `infra/persistence/sqlite/_repos/<name>.py`

```python
from everos.infra.persistence.sqlite._base.repository import RepoBase

from ..tables import EpisodeState


class EpisodeStateRepo(RepoBase[EpisodeState]):
    """Rows of ``episode_state``."""

    model = EpisodeState
```

The session factory comes in through `RepoBase.__init__`. Do **not** create
a module-level instance: `make check-singletons` rejects one, and the
repository would be bound to no database.

### 4.4 Re-export + wire — `repos/__init__.py`, `repositories.py`

```python
# repos/__init__.py
from .episode import EpisodeStateRepo as EpisodeStateRepo
```

Add a field to `SqliteRepos` (`sqlite/repositories.py`) and build it in
`SqliteRepos.over(database)`; the bootstrap hands that bundle to whatever
needs the table.

---

## 5. (Optional) Derived index — 4 steps

Skip this section if the kind doesn't need vector / BM25 / hybrid retrieval.

### 5.1 Record declaration — `infra/persistence/records/<name>.py`

```python
from typing import ClassVar

from everos.infra.persistence.records import BaseLanceTable, Vector


class EpisodeIndex(BaseLanceTable):
    TABLE_NAME: ClassVar[str] = "episode_index"
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")

    id: str
    app_id: str = "default"
    project_id: str = "default"
    entry_id: str
    text: str
    tags: list[str]
    vector: Vector(1024) | None = None  # type: ignore[valid-type]
```

`TABLE_NAME` / `BM25_FIELDS` / `PRIMARY_KEY` are the backend-neutral
declarations every index backend reads (`IndexRecordMeta`). `PRIMARY_KEY`
has **no default**: `check_primary_keys` refuses to start, and LanceDB
`upsert` refuses to write, a table that does not declare one. `id` is unique
only within `(app_id, project_id)`, so every business table declares the
three-column key — LanceDB merges on it, Milvus keys the collection on a
composed `__pk` and rejects key components containing `/`. `Vector(N)` must
match your embedding dimension.

### 5.2 Re-export — `records/__init__.py`

```python
from .episode import EpisodeIndex as EpisodeIndex
```

### 5.3 Repo — `infra/persistence/lancedb/_repos/<name>.py`

```python
from everos.infra.persistence.lancedb._base.repository import LanceRepoBase

from ...records.episode import EpisodeIndex


class _EpisodeIndexRepo(LanceRepoBase[EpisodeIndex]):
    schema = EpisodeIndex
```

The table comes from the `LanceDbConnection` passed as `source=`; no
module-level instance (`make check-singletons` rejects one).

`upsert(records)` merges on the declared key and is the only write; look a
row up with `find_one_where(primary_key(app_id, project_id, row_id))`.

### 5.4 Wire — `index/repositories.py` + each backend

Add the record to `BUSINESS_SCHEMAS` and a field to `IndexRepositories`
(`infra/persistence/index/repositories.py`), then build the repository in
`LanceIndexBackend.__init__` (`_EpisodeIndexRepo(source=connection)`) and in
`MilvusIndexBackend.__init__`. Business code reaches it as
`index.repositories.<field>`, through the `DerivedIndex` the bootstrap
builds.

---

## 6. Single-file markdown strategy

When the new memory kind is one evolving document per owner (like
`user.md`), mix `ProfilePathMixin` in before the scope mixin and declare
`PROFILE_FILENAME` (see `UserProfileFrontmatter` in `_mds/profile.py`); a
file under a sub-directory overrides `path_glob()` instead (see
`GroupMemberProfileFrontmatter`, `groups/<g>/members/<m>.md`). The whole
file is rewritten by its store (`MarkdownProfileStore` in
`infra/persistence/markdown/stores.py`), not by a `BaseDailyWriter`.

This strategy does **not** use entry markers; its frontmatter schema
does not need `ENTRY_ID_PREFIX` (only `id` / `type` / `schema_version` plus
the scope mixin fields).

---

## 7. Verification checklist

- [ ] `make lint` — ruff + import-linter clean
- [ ] `make test` — existing manager / lifespan / writer tests still pass
- [ ] When the kind crosses **multiple** layers:
  - [ ] markdown entry id is the join key for the sqlite / lancedb rows
  - [ ] business code reads only via the injected repository (no raw
        engine access in service / memory)
  - [ ] cascade daemon (when it lands) can rebuild sqlite / lancedb
        from md alone — keep md as the truth

Tests by layer:

| Tests for | Location |
|---|---|
| Markdown frontmatter schema | `tests/unit/test_infra/test_markdown/test_mds/` |
| Markdown business appender | `tests/unit/test_infra/test_markdown/test_writers/` |
| SQLite RepoBase logic | `tests/unit/test_core/test_persistence/test_sqlite/` |
| SQLite database object | `tests/unit/test_infra/test_sqlite/` |
| LanceDB LanceRepoBase logic | `tests/unit/test_core/test_persistence/test_lancedb/` |
| LanceDB connection object | `tests/unit/test_infra/test_lancedb/` |

## 8. Common pitfalls

| Mistake | Symptom | Fix |
|---|---|---|
| Forgot `ENTRY_ID_PREFIX` / `DIR_NAME` / `FILE_PREFIX` on a daily-log schema | `BaseDailyWriter.__init__` raises `TypeError` | Add all three ClassVars |
| Same `ENTRY_ID_PREFIX` on user + group variants | A group whose id equals a user's id shares that user's index row ids | Use distinct prefixes (e.g. `ep` vs `gep`) |
| Skipped one of the four files (schema / writer / table / repo) | One side silently absent | Re-export both/all from each `__init__.py` |
| `Vector(N)` mismatched with embedding dim | LanceDB raises on insert | Make `N` exactly match the model output |
| Imported `MemoryLayout` from a writer (infra) | `import-linter` fails (`infra → memory` reverse dep) | Use `MemoryRoot` (in `infra.persistence`) and let the schema's ClassVars drive paths |
| Hand-rolling `datetime.now()` instead of `today_with_timezone()` | Day-boundary drift across timezones | Always go through `everos.core.utils.datetime` |

## Background

- Architecture: [../../rules/architecture.md](../../rules/architecture.md)
- `__init__.py` re-export rules: [../../rules/init-py-and-reexport.md](../../rules/init-py-and-reexport.md)
- Async programming: [../../rules/async-programming.md](../../rules/async-programming.md)
- Datetime handling: [../../rules/datetime-handling.md](../../rules/datetime-handling.md)
