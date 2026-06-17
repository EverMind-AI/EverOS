"""Private helpers shared across the search e2e tests.

* :func:`pick_query_seeds` — scans the session corpus's
  ``.atomic_facts/`` md files and returns a list of
  ``(owner_id, fact_text)`` tuples to use as deterministic search
  queries. Bootstrapping queries off the corpus's own extraction
  output gives us a closed-loop correctness signal — what was
  written should be findable.

* :func:`assert_recall` — the canonical "this search returned at
  least one sensible hit for ``owner``" assertion bundle. Used by
  the keyword / vector / hybrid recall tests so the assertion logic
  is in one place.

* :func:`flatten_hits` — collapses ``SearchData``'s four scored result
  arrays into one ``(owner_id, score, text)`` tuple list for relevance
  checks.

The helpers do **not** hardcode topical keywords ("hiking" / "work")
— they are derived from what the pipeline produced. This keeps the
suite stable across LLM-driven boundary-cut variance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

# Cap how many fact strings we sample per call — running every test
# against every fact would blow the LLM rerank budget.
_DEFAULT_SEED_LIMIT = 3

# Tokenise on word characters; lowercase; drop short tokens that carry
# no signal for the "content overlap" check.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MIN_TOKEN_LEN = 3
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "was",
        "has",
        "have",
        "are",
        "but",
        "from",
        "you",
        "she",
        "her",
        "his",
        "him",
        "they",
        "them",
        "their",
    }
)


# ── Query seed extraction ───────────────────────────────────────────────


def pick_query_seeds(
    memory_root: Path,
    *,
    limit: int = _DEFAULT_SEED_LIMIT,
) -> list[tuple[str, str]]:
    """Sample ``(owner_id, fact_text)`` tuples from atomic_facts md files.

    Walks ``users/<owner>/.atomic_facts/atomic_fact-*.md`` and parses
    the ``## Fact\\n<text>`` sections inside each daily-log entry.
    Returns deterministic seeds (insertion order of ``rglob`` is
    sort-stable thanks to the explicit ``sorted`` call) so a flaky
    test surfaces a real regression, not query-rotation variance.

    Raises:
        AssertionError: if no facts were extracted — that's a fixture
            failure, not a test failure, and should fail loudly.
    """
    seeds: list[tuple[str, str]] = []
    users_dir = memory_root / "default_app" / "default_project" / "users"
    if not users_dir.is_dir():
        raise AssertionError(f"expected {users_dir} to exist after ingest")

    for owner_dir in sorted(users_dir.iterdir()):
        if not owner_dir.is_dir():
            continue
        facts_dir = owner_dir / ".atomic_facts"
        if not facts_dir.is_dir():
            continue
        for md in sorted(facts_dir.rglob("*.md")):
            for fact in _extract_fact_sections(md):
                if fact:
                    seeds.append((owner_dir.name, fact))
                    if len(seeds) >= limit:
                        return seeds
    if not seeds:
        raise AssertionError(
            f"no atomic_fact md entries under {users_dir} — pipeline did "
            "not produce any facts; cannot bootstrap search queries"
        )
    return seeds


def _extract_fact_sections(md: Path) -> list[str]:
    """Return every ``### Fact`` section body in a daily-log md file.

    Daily-log entries are ``## <entry-id>`` blocks; the labelled body
    sections inside an entry are h3 (``### Fact``, ``### Foresight``,
    …). We scan linearly for ``### Fact`` and collect lines until the
    next heading at any level or the end-of-entry marker.
    """
    body = md.read_text(encoding="utf-8")
    sections: list[str] = []
    in_fact = False
    buf: list[str] = []
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("### Fact"):
            if in_fact:
                sections.append("\n".join(buf).strip())
            in_fact = True
            buf = []
            continue
        # Any subsequent heading or entry-end marker closes the section.
        if in_fact and (stripped.startswith("#") or stripped.startswith("<!-- /entry")):
            sections.append("\n".join(buf).strip())
            in_fact = False
            buf = []
            continue
        if in_fact:
            buf.append(line)
    if in_fact:
        sections.append("\n".join(buf).strip())
    return [s for s in sections if s]


# ── Response flattening + assertions ────────────────────────────────────


def flatten_hits(data: dict[str, Any]) -> list[tuple[str | None, float, str]]:
    """Collapse the four scored arrays into ``(owner_id, score, text)``.

    Stable shape across track-kinds so the recall / partition tests
    don't have to branch. Episodes / profiles carry ``user_id`` on the
    item; cases / skills carry ``agent_id`` — both project to the
    generic ``owner`` slot here. ``owner`` may be ``None`` for profile
    hits where the owner is implicit.
    """
    out: list[tuple[str | None, float, str]] = []
    for ep in data.get("episodes", []):
        out.append(
            (
                ep.get("user_id"),
                float(ep.get("score") or 0.0),
                ep.get("episode") or ep.get("summary") or ep.get("subject") or "",
            )
        )
    for pf in data.get("profiles", []):
        out.append(
            (
                pf.get("user_id"),
                float(pf.get("score") or 0.0),
                str(pf.get("profile_data") or ""),
            )
        )
    for cs in data.get("agent_cases", []):
        out.append(
            (
                cs.get("agent_id"),
                float(cs.get("score") or 0.0),
                cs.get("approach") or cs.get("task_intent") or "",
            )
        )
    for sk in data.get("agent_skills", []):
        out.append(
            (
                sk.get("agent_id"),
                float(sk.get("score") or 0.0),
                sk.get("content") or sk.get("description") or "",
            )
        )
    return out


async def assert_recall(
    client: httpx.AsyncClient,
    *,
    owner_id: str,
    query: str,
    method: str,
    min_score: float = 0.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Hit ``/search`` and lock the four standard recall invariants.

    1. **Status** 200 — the route compiled.
    2. **Existence** — ``total >= 1`` across the four scored arrays.
    3. **Owner partition** — every non-``None`` ``owner_id`` matches
       the queried owner. Profile hits may carry ``None`` so they're
       skipped from the check.
    4. **Score sanity** — the top-scored hit clears ``min_score``.

    Returns the parsed response body so the caller can layer
    case-specific assertions on top.
    """
    resp = await client.post(
        "/api/v1/memory/search",
        json={
            "user_id": owner_id,
            "query": query,
            "method": method,
            "top_k": top_k,
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    hits = flatten_hits(body["data"])
    assert hits, (
        f"no hits for owner={owner_id} query={query!r} method={method} — "
        f"recall is broken"
    )
    for hit_owner, _score, _text in hits:
        if hit_owner is not None:
            assert hit_owner == owner_id, (
                f"partition leak: got owner={hit_owner!r} when querying {owner_id!r}"
            )
    top_score = max(score for _o, score, _t in hits)
    assert top_score >= min_score, (
        f"top hit score {top_score:.3f} < min {min_score} for method={method}"
    )
    return body


# ── Token utilities (for content-overlap checks) ────────────────────────


def query_tokens(query: str) -> set[str]:
    """Lowercase content tokens worth checking for overlap in hit text."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(query)
        if len(t) >= _MIN_TOKEN_LEN and t.lower() not in _STOPWORDS
    }


def content_tokens_in_order(query: str) -> list[str]:
    """Content tokens in original document order, dedup'd by first occurrence.

    Used by the keyword test: the project's BM25 tokenizer (jieba) is
    Chinese-first and degrades to near-zero recall on single short
    English tokens. Multi-token phrases recall well in practice, so
    keyword queries are built by concatenating consecutive content
    tokens from the source fact rather than sorting alphabetically.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in _TOKEN_RE.findall(query):
        low = t.lower()
        if len(t) >= _MIN_TOKEN_LEN and low not in _STOPWORDS and low not in seen:
            seen.add(low)
            out.append(low)
    return out
