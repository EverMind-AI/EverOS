"""BM25 recall settles score ties by id, so a repeated search recalls one list.

LanceDB's full-text search returns tied rows in no fixed order and reports a
row's score with ~1e-6 of noise from call to call, so a plain ``limit`` kept a
different subset of a tie group at the cut on every call. The parity check
against the exp tree (2026-11-10, deviation R3) found the episode and picture
recallers here doing exactly that, while the exp tree's ``fts_rows`` -- and its
2.0 picture port -- return the same list every time. The exp tree applies the
same rule to every BM25 recall, so these tests pin it (:func:`stable_sparse_top`)
on both recallers: fetch past the limit until the tie group at the cut is
complete, order by (score to 5 decimals, id), cut.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.persistence.index import all_of, eq
from everos.infra.persistence.index.adapters import EpisodeIndexAdapter
from everos.infra.persistence.lancedb import Episode as EpisodeRow
from everos.infra.persistence.lancedb import ParentType
from everos.memory.models import ImageMemory
from everos.memory.ports import Filter, MemoryRef, SearchHit, Stored
from everos.memory.search.recall.base import RecallerDeps, stable_sparse_top
from everos.memory.search.recall.episode import EpisodeRecaller
from everos.memory.search.recall.image_memory import IMAGE_KIND, ImageMemoryRecaller
from tests.fakes.storage import Storage

from ._records import scope, stored_episode

_WHERE = eq("owner_id", "alice")


class _Tokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


def _stored_picture(entry_id: str) -> Stored[ImageMemory]:
    image_id = entry_id.encode().hex().ljust(64, "0")[:64]
    return Stored(
        ref=MemoryRef(
            id=f"alice_{entry_id}",
            entry_id=entry_id,
            md_path="users/alice/images/image.md",
        ),
        scope=scope(),
        item=ImageMemory(
            owner_id="alice",
            description=f"picture {entry_id}",
            timestamp=1_767_225_600_000,
            image_id=image_id,
            original_relpath=f"figures/{image_id[:2]}/{image_id}.png",
            media_type="png",
            width=64,
            height=64,
        ),
    )


class _NoisyIndex:
    """An index port that behaves like LanceDB FTS where it matters.

    The top ``limit`` hits by score -- but tied hits come back in a random order
    and every score carries ~1e-6 of noise, so a plain cut keeps a different
    subset of a tie group per call. Every ``limit`` asked for is recorded.
    """

    def __init__(
        self,
        scores: dict[str, float],
        seed: int,
        *,
        stored: Callable[[str], Stored[Any]] = stored_episode,
    ) -> None:
        self._scores = scores
        self._rng = random.Random(seed)
        self._stored = stored
        self.limits: list[int] = []

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[Any]]:
        self.limits.append(limit)
        rows = [
            (key, score + self._rng.uniform(-1e-6, 1e-6))
            for key, score in self._scores.items()
        ]
        self._rng.shuffle(rows)
        rows.sort(key=lambda row: -round(row[1], 4))  # stable: ties stay shuffled
        return [
            SearchHit(stored=self._stored(key), score=score)
            for key, score in rows[:limit]
        ]


def _scores() -> dict[str, float]:
    # 10 distinct leaders, then a 60-row tie group straddling every cut below.
    out = {f"lead{i:02d}": 20.0 - i for i in range(10)}
    out.update({f"tie{i:02d}": 6.810414 for i in range(60)})
    out.update({f"tail{i:02d}": 1.0 + i / 100 for i in range(30)})
    return out


def _episodes(index: _NoisyIndex) -> EpisodeRecaller:
    return EpisodeRecaller(RecallerDeps(tokenizer=_Tokenizer()), index)  # type: ignore[arg-type]


def _pictures(index: _NoisyIndex) -> ImageMemoryRecaller:
    return ImageMemoryRecaller(RecallerDeps(tokenizer=_Tokenizer()), index)  # type: ignore[arg-type]


async def test_episodes_recall_the_same_list_whatever_the_engine_returns() -> None:
    runs = []
    for seed in range(8):
        result = await _episodes(_NoisyIndex(_scores(), seed)).sparse_recall(
            "marathon", _WHERE, limit=50
        )
        runs.append([c.id for c in result.candidates])

    assert all(run == runs[0] for run in runs)
    expected = [f"alice_lead{i:02d}" for i in range(10)]
    # The cut inside the tie group keeps the lowest ids, not whichever came first.
    expected += [f"alice_tie{i:02d}" for i in range(40)]
    assert runs[0] == expected


async def test_pictures_recall_the_same_list_whatever_the_engine_returns() -> None:
    runs = []
    for seed in range(8):
        index = _NoisyIndex(_scores(), seed, stored=_stored_picture)
        result = await _pictures(index).sparse_recall("sign", _WHERE, limit=50)
        runs.append([c.id for c in result.candidates])
        assert {c.metadata["kind"] for c in result.candidates} == {IMAGE_KIND}

    assert all(run == runs[0] for run in runs)
    assert runs[0][10:] == [f"alice_tie{i:02d}" for i in range(40)]


async def test_the_fetch_widens_until_the_tie_group_is_complete() -> None:
    index = _NoisyIndex(_scores(), seed=0)

    result = await _episodes(index).sparse_recall("marathon", _WHERE, limit=20)

    assert [c.id for c in result.candidates] == [
        *(f"alice_lead{i:02d}" for i in range(10)),
        *(f"alice_tie{i:02d}" for i in range(10)),
    ]
    # 52 rows still end inside the 60-row tie group (ranks 10-69); 104 reach past it.
    assert index.limits == [52, 104]


async def test_fewer_matches_than_the_fetch_come_back_in_score_then_id_order() -> None:
    index = _NoisyIndex({"b": 2.0, "a": 2.0, "c": 3.0}, seed=1)

    result = await _episodes(index).sparse_recall("marathon", _WHERE, limit=10)

    assert [c.id for c in result.candidates] == ["alice_c", "alice_a", "alice_b"]
    assert index.limits == [42]


async def test_the_raw_score_is_kept_on_the_candidate() -> None:
    """Rounding orders the hits; it does not rewrite the score RRF and traces see."""
    index = _NoisyIndex({"a": 2.123456789}, seed=0)

    (cand,) = (
        await _episodes(index).sparse_recall("marathon", _WHERE, limit=5)
    ).candidates

    assert cand.score == pytest.approx(2.123456789, abs=2e-6)
    assert cand.score != round(cand.score, 5)


async def test_a_zero_limit_queries_nothing() -> None:
    calls: list[int] = []

    async def search(n: int) -> list[SearchHit[Any]]:
        calls.append(n)
        return []

    assert await stable_sparse_top(search, 0) == []
    assert calls == []


async def test_widening_gives_up_at_the_cap() -> None:
    """A tie group bigger than 32x the limit is cut where the cap leaves it."""
    index = _NoisyIndex({f"t{i:04d}": 1.0 for i in range(400)}, seed=3)

    result = await _episodes(index).sparse_recall("marathon", _WHERE, limit=5)

    assert index.limits == [37, 74, 148, 160]
    assert len(result.candidates) == 5


# ── on the real engine ─────────────────────────────────────────────────────


def _episode_row(n: int) -> EpisodeRow:
    return EpisodeRow(
        id=f"alice_ep_{n:02d}",
        entry_id=f"ep_{n:02d}",
        owner_id="alice",
        owner_type="user",
        session_id="sess_1",
        timestamp=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        parent_type=ParentType.MEMCELL.value,
        parent_id="mc_test",
        sender_ids=["alice"],
        episode="harbor marathon",
        episode_tokens="harbor marathon",
        md_path="users/alice/episodes/episode-2026-01-01.md",
        content_sha256="x" * 64,
        vector=[0.0] * 1024,
    )


@pytest.fixture(autouse=True)
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))


async def test_lancedb_ties_are_cut_by_id(storage: Storage) -> None:
    """25 identical episodes, written highest id first: the cut keeps the lowest
    ids, on every call, however LanceDB orders the tie."""
    await storage.repos.episode.upsert([_episode_row(n) for n in reversed(range(25))])
    table = await storage.lance.table(EpisodeRow.TABLE_NAME, EpisodeRow)
    await table.optimize()
    recaller = EpisodeRecaller(
        RecallerDeps(tokenizer=_Tokenizer()),
        EpisodeIndexAdapter(storage.repos.episode),
    )
    where = all_of(eq("owner_id", "alice"), eq("owner_type", "user"))

    runs = [
        [
            c.id
            for c in (
                await recaller.sparse_recall("harbor", where, limit=10)
            ).candidates
        ]
        for _ in range(3)
    ]

    assert runs == [[f"alice_ep_{n:02d}" for n in range(10)]] * 3
