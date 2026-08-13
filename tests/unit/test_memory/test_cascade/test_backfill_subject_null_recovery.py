"""Round-2 finding #1: subject-only embed failure must not silently
corrupt the ``subject_vector`` column.

Round-1 fixed the wall-clock regression on Episode's dual embed
(finding #10) by wrapping ``_embed_primary_batch`` and
``_embed_subject_batch`` in ``asyncio.gather``. But when only the
subject leg failed (``_embed_subject_batch`` catches Exception and
returns ``{}``), the primary vector still got written and
``rows_processed`` still incremented — leaving the row with
``vector`` populated but ``subject_vector`` NULL. Because the scan
filter checked only ``vector IS NULL``, subsequent backfill runs
could not detect the residue: the row was permanently keyword-only
for its subject half.

Round-2 widens the Episode scan filter to
``vector IS NULL OR subject_vector IS NULL`` (see :func:`_null_filter`),
and rewrites ``_backfill_table`` to embed only the side each row
actually needs — a re-run picks up the residue and re-embeds just the
missing side.

This file pins both invariants:

- Subject-only failure leaves the row visible to the next scan.
- A batch mixing primary-only, subject-only, and both-null rows
  ends with every row fully embedded after re-running.
"""

from __future__ import annotations

from typing import Any

from everos.memory.cascade._backfill import (
    NullBackfillPresenter,
    _backfill_table,
    _extract_row,
    _null_filter,
    _NullVectorRow,
    _TableBacklog,
    _TableSpec,
)


class _EpisodeLikeSchema:
    """Stand-in schema whose ``TABLE_NAME`` matches
    :attr:`Episode.TABLE_NAME` so :func:`_null_filter` returns the
    widened ``OR subject_vector IS NULL`` clause."""

    TABLE_NAME = "episode"


class _NonSubjectSchema:
    """Non-Episode schema — filter should stay ``vector IS NULL``."""

    TABLE_NAME = "atomic_fact"


class _RecordingRepo:
    """Repo double that records every ``update`` call so tests can
    assert on the exact ``{col: value}`` shape written back."""

    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, Any], str]] = []

    async def update(self, values: dict[str, Any], *, where: str) -> None:
        self.updates.append((values, where))


class _SubjectFailingProvider:
    """First ``embed_batch`` call succeeds (primary batch); second
    raises (subject batch). Mirrors the fault mode round-1 missed."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("simulated subject embed failure")
        return [[1.0, 2.0, 3.0] for _ in texts]


class _AllSucceedProvider:
    """Both sides succeed — used in the recovery-run assertions."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[9.9] * 3 for _ in texts]


def _episode_spec(repo: _RecordingRepo) -> _TableSpec:
    return _TableSpec(
        schema=_EpisodeLikeSchema,  # type: ignore[arg-type]
        repo=repo,  # type: ignore[arg-type]
        text_of=lambda r: r["episode"],
        subject_of=lambda r: r.get("subject") or None,
    )


async def test_subject_only_failure_leaves_row_in_backlog() -> None:
    """Subject-side embed fails → row's ``vector`` is written, its
    ``subject_vector`` is NOT written. The widened scan filter for
    Episode picks the residue up on the next scan.
    """
    repo = _RecordingRepo()
    spec = _episode_spec(repo)
    rows = [
        _NullVectorRow(
            id=f"r{i}",
            text=f"episode {i}",
            subject_text=f"subject {i}",
            tokens=5,
            needs_primary=True,
            needs_subject=True,
        )
        for i in range(3)
    ]
    backlog = _TableBacklog(spec=spec, rows=rows)

    provider = _SubjectFailingProvider()
    result = await _backfill_table(backlog, provider, presenter=NullBackfillPresenter())  # type: ignore[arg-type]

    # Every row's primary vector landed on disk exactly once, and no
    # ``subject_vector`` field was written for any row.
    assert len(repo.updates) == len(rows)
    for values, _where in repo.updates:
        assert "vector" in values
        assert "subject_vector" not in values
    # Subject side failed for every row — round-4 review J7: rows with
    # any needed side still NULL count as ``rows_failed`` so the CLI
    # exit code escalates to COMPLETED_WITH_FAILURES (4). The widened
    # scan filter picks them up next run to retry the missing side.
    assert result.rows_processed == 0
    assert result.rows_failed == len(rows)

    # The widened Episode scan filter would still match these rows
    # since their subject_vector remains NULL.
    assert _null_filter(spec) == "vector IS NULL OR subject_vector IS NULL"

    # Simulate a fresh scan encountering the residue: primary vector
    # is now populated (from the last write), subject_vector is None.
    tokenizer = _FakeTokenizer()
    raw = {
        "id": "r0",
        "episode": "episode 0",
        "subject": "subject 0",
        "vector": [1.0, 2.0, 3.0],
        "subject_vector": None,
    }
    extracted = _extract_row(raw, spec, tokenizer)
    assert isinstance(extracted, _NullVectorRow)
    assert extracted.needs_primary is False
    assert extracted.needs_subject is True


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


async def test_orthogonal_partial_states_all_recover() -> None:
    """Mixed batch: fully-null, primary-only-null, subject-only-null.
    Each row gets its missing side(s) embedded — no wasted embed on
    the side that's already fine.
    """
    repo = _RecordingRepo()
    spec = _episode_spec(repo)
    rows = [
        # fully null — needs both
        _NullVectorRow(
            id="both",
            text="e_both",
            subject_text="s_both",
            tokens=5,
            needs_primary=True,
            needs_subject=True,
        ),
        # primary already set — needs subject only
        _NullVectorRow(
            id="subj_only",
            text="e_subj",
            subject_text="s_subj",
            tokens=5,
            needs_primary=False,
            needs_subject=True,
        ),
        # subject already set — needs primary only
        _NullVectorRow(
            id="prim_only",
            text="e_prim",
            subject_text="s_prim",
            tokens=5,
            needs_primary=True,
            needs_subject=False,
        ),
    ]
    backlog = _TableBacklog(spec=spec, rows=rows)

    provider = _AllSucceedProvider()
    result = await _backfill_table(backlog, provider, presenter=NullBackfillPresenter())  # type: ignore[arg-type]

    assert result.rows_processed == 3
    assert result.rows_failed == 0

    # Recover per-id updates by parsing the where clause
    # (``id = 'xxx'``) — the repo double preserves call order and shape.
    updates_by_id: dict[str, dict[str, Any]] = {}
    for values, where in repo.updates:
        row_id = where.split("'")[1]
        updates_by_id[row_id] = values

    assert set(updates_by_id) == {"both", "subj_only", "prim_only"}
    assert "vector" in updates_by_id["both"]
    assert "subject_vector" in updates_by_id["both"]
    assert "vector" not in updates_by_id["subj_only"]
    assert "subject_vector" in updates_by_id["subj_only"]
    assert "vector" in updates_by_id["prim_only"]
    assert "subject_vector" not in updates_by_id["prim_only"]


def test_null_filter_widens_only_for_episode() -> None:
    """Round-2 finding #1 mapping: Episode gets the OR clause; other
    tables keep the primary-only filter (their tables don't carry a
    ``subject_vector`` column, so referencing it would blow up)."""
    ep_repo = _RecordingRepo()
    ep_spec = _episode_spec(ep_repo)
    assert _null_filter(ep_spec) == "vector IS NULL OR subject_vector IS NULL"

    fact_spec = _TableSpec(
        schema=_NonSubjectSchema,  # type: ignore[arg-type]
        repo=_RecordingRepo(),  # type: ignore[arg-type]
        text_of=lambda r: r["fact"],
    )
    assert _null_filter(fact_spec) == "vector IS NULL"


def test_extract_row_skips_legitimate_null_subject() -> None:
    """A row where ``subject`` is legitimately absent and both
    ``subject_vector`` NULL + ``vector`` populated is NOT a backlog
    item — ``_extract_row`` returns the ``_ROW_SKIPPED`` sentinel."""
    from everos.memory.cascade._backfill import _ROW_SKIPPED

    spec = _episode_spec(_RecordingRepo())
    raw = {
        "id": "r0",
        "episode": "just a body",
        "subject": None,
        "vector": [1.0, 2.0, 3.0],
        "subject_vector": None,
    }
    out = _extract_row(raw, spec, _FakeTokenizer())
    assert out is _ROW_SKIPPED
