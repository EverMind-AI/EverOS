---
name: add-memory-kind
description: Add a new business memory kind end-to-end. Pick the storage combination (Markdown / SQLite / LanceDB), pick the markdown strategy (daily-log / skill-named / single-file), then wire up the schema(s), repo(s), and writer(s).
---

# /add-memory-kind — Add a new business memory kind

## When to invoke

Adding a new persisted business entity (Episode, Case, Skill, AtomicFact,
Foresight, Profile, or something custom). Multiple storage layers may be
involved; this skill walks the decision then the wiring.

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
| **md only** | scratch notes / dump bins | text-of-truth, no index needed |
| **md + lancedb** | episode / memcell / case | text-of-truth + semantic retrieval |
| **md + sqlite** | profile / playbook / soul.md state | text-of-truth + structured state to query |
| **md + sqlite + lancedb** | full-blown business records | when you need *both* transactional state AND retrieval |
| **sqlite only** | audit log / task queue / LSN watermark | system state, never user-facing |
| **lancedb only** | rare; usually you still want md | derived embeddings without text-of-truth |

Rule of thumb: **markdown is the truth**; sqlite and lancedb are derived
indexes that can be rebuilt from md. Drop md only when the kind has no
human-readable form (pure system state).

## 2. Pick the markdown storage strategy (if md is in your combo)

Three strategies — declared in the EverOS Markdown First spec:

| Strategy | Filename | Mutation | Examples |
|---|---|---|---|
| **Daily-log append** | `<prefix>-YYYY-MM-DD.md` | append entries | memcell / episode / case / atomic_fact / foresight |
| **Skill-named in-place** | `skill_<name>.md` | overwrite the file | skills (procedural memory) |
| **Single-file rewrite** | `user.md` / `agent.md` / `soul.md` / `behaviors.md` / `tools.md` | overwrite the file | profiles / playbooks |

This skill currently has a **complete recipe for daily-log append**.
Skill-named and single-file recipes are sketched at the bottom — their
base writers (`BaseSkillWriter` / `BaseProfileWriter`) land later in the
project; until then build a thin wrapper over `MarkdownWriter`
directly.

---

## 3. Markdown daily-log: 4 steps

### 3.1 Frontmatter schema — `infra/persistence/markdown/mds/<name>.py`

```python
"""Episode daily-log frontmatter."""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from everos.core.persistence.markdown import UserScopedFrontmatter


class UserEpisodeDailyFrontmatter(UserScopedFrontmatter):
    """``users/<u>/episodes/episode-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "ep"
    DIR_NAME: ClassVar[str] = "episodes"
    FILE_PREFIX: ClassVar[str] = "episode"

    type: Literal["user_episode_daily"] = "user_episode_daily"
    date: _dt.date
    entry_count: int = 0
    last_appended_at: _dt.datetime | None = None
```

For agent-track kinds subclass `AgentScopedFrontmatter` instead. If
user-track and agent-track share a kind name (e.g. `memcell`), give
each a **distinct** `ENTRY_ID_PREFIX` (e.g. `umc` vs `amc`) so reverse
lookup is unambiguous.

### 3.2 Re-export — `mds/__init__.py`

```python
from .episode import UserEpisodeDailyFrontmatter as UserEpisodeDailyFrontmatter
```

### 3.3 Business writer — `infra/persistence/markdown/writers/<name>.py`

```python
"""Episode appender."""

from __future__ import annotations

from pathlib import Path

from everos.core.persistence import MarkdownReader

from ..mds import UserEpisodeDailyFrontmatter
from .base import BaseDailyWriter


class UserEpisodeAppender(BaseDailyWriter):
    schema = UserEpisodeDailyFrontmatter

    # OPTIONAL: override the count strategy. Default is len(entries);
    # override to trust the frontmatter field instead.
    def _current_count(self, path: Path) -> int:
        if not path.exists():
            return 0
        return MarkdownReader.read(path).frontmatter.get("entry_count", 0)
```

### 3.4 Re-export — `writers/__init__.py`

```python
from .episode import UserEpisodeAppender as UserEpisodeAppender
```

### Done — usage

```python
from everos.infra.persistence.markdown.writers import UserEpisodeAppender

appender = UserEpisodeAppender(memory_root)
eid = appender.append("u_jason", "I went to the doctor today.")
# → users/u_jason/episodes/episode-<today>.md
# → entry markers carry an auto-generated EntryId (e.g. ep_20260507_001)
```

---

## 4. (Optional) SQLite table — 4 steps

Skip this section if the kind doesn't need structured state beyond markdown.

### 4.1 Schema — `infra/persistence/sqlite/tables/<name>.py`

```python
from everos.core.persistence.sqlite import BaseTable, Field


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

### 4.3 Repo — `infra/persistence/sqlite/repos/<name>.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase

from ..sqlite_manager import get_session_factory
from ..tables import EpisodeState


class _EpisodeStateRepo(RepoBase[EpisodeState]):
    model = EpisodeState

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()


episode_state_repo = _EpisodeStateRepo()
```

### 4.4 Re-export — `repos/__init__.py`

```python
from .episode import episode_state_repo as episode_state_repo
```

---

## 5. (Optional) LanceDB index — 4 steps

Skip this section if the kind doesn't need vector / BM25 / hybrid retrieval.

### 5.1 Schema — `infra/persistence/lancedb/tables/<name>.py`

```python
from everos.core.persistence.lancedb import BaseLanceTable, Vector


class EpisodeIndex(BaseLanceTable):
    entry_id: str
    text: str
    tags: list[str]
    vector: Vector(384)  # type: ignore[valid-type]
```

`Vector(N)` must match your embedding dimension.

### 5.2 Re-export — `tables/__init__.py`

```python
from .episode import EpisodeIndex as EpisodeIndex
```

### 5.3 Repo — `infra/persistence/lancedb/repos/<name>.py`

```python
from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceRepoBase

from ..lancedb_manager import get_table
from ..tables import EpisodeIndex


class _EpisodeIndexRepo(LanceRepoBase[EpisodeIndex]):
    schema = EpisodeIndex
    table_name = "episode_index"

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.table_name, self.schema)


episode_index_repo = _EpisodeIndexRepo()
```

### 5.4 Re-export — `repos/__init__.py`

```python
from .episode import episode_index_repo as episode_index_repo
```

---

## 6. (Future) Skill-named & single-file markdown strategies

When the new memory kind needs:

- **skill-named** files (one file per named skill, in-place rewrite) — wait
  for `BaseSkillWriter`, or use `MarkdownWriter.write_markdown` directly
  with a thin wrapper.
- **single-file rewrite** (one fixed file like `user.md`) — wait for
  `BaseProfileWriter`, same fallback.

These strategies do **not** use entry markers; their frontmatter schema
does not need `ENTRY_ID_PREFIX` (only `id` / `type` / `schema_version` plus
the scope mixin fields).

---

## 7. Verification checklist

- [ ] `make lint` — ruff + import-linter clean
- [ ] `make test` — existing manager / lifespan / writer tests still pass
- [ ] When the kind crosses **multiple** layers:
  - [ ] markdown entry id is the join key for the sqlite / lancedb rows
  - [ ] business code reads only via the repo singleton (no raw engine
        access in service / memory)
  - [ ] cascade daemon (when it lands) can rebuild sqlite / lancedb
        from md alone — keep md as the truth

Tests by layer:

| Tests for | Location |
|---|---|
| Markdown frontmatter schema | `tests/unit/test_infra/test_markdown/test_mds/` |
| Markdown business appender | `tests/unit/test_infra/test_markdown/test_writers/` |
| SQLite RepoBase logic | `tests/unit/test_core/test_persistence/test_sqlite/` |
| SQLite manager / lifespan | `tests/unit/test_infra/test_sqlite/` |
| LanceDB LanceRepoBase logic | `tests/unit/test_core/test_persistence/test_lancedb/` |
| LanceDB manager / lifespan | `tests/unit/test_infra/test_lancedb/` |

## 8. Common pitfalls

| Mistake | Symptom | Fix |
|---|---|---|
| Forgot `ENTRY_ID_PREFIX` / `DIR_NAME` / `FILE_PREFIX` on a daily-log schema | `BaseDailyWriter.__init__` raises `TypeError` | Add all three ClassVars |
| Same `ENTRY_ID_PREFIX` on user + agent variants | `MemoryLayout.locate_for_entry` collision error | Use distinct prefixes (e.g. `umc` vs `amc`) |
| Imported `RepoBase` from `infra.persistence.sqlite` | `ImportError` | Lives in `core.persistence.sqlite` (moved earlier) |
| Skipped one of the four files (schema / writer / table / repo) | One side silently absent | Re-export both/all from each `__init__.py` |
| `Vector(N)` mismatched with embedding dim | LanceDB raises on insert | Make `N` exactly match the model output |
| Imported `MemoryLayout` from a writer (infra) | `import-linter` fails (`infra → memory` reverse dep) | Use `MemoryRoot` (in core) and let the schema's ClassVars drive paths |
| Hand-rolling `datetime.now()` instead of `today_with_timezone()` | Day-boundary drift across timezones | Always go through `everos.component.utils.datetime` |

## Background

- Architecture: [../../rules/architecture.md](../../rules/architecture.md)
- `__init__.py` re-export rules: [../../rules/init-py-and-reexport.md](../../rules/init-py-and-reexport.md)
- Async programming: [../../rules/async-programming.md](../../rules/async-programming.md)
- Datetime handling: [../../rules/datetime-handling.md](../../rules/datetime-handling.md)
