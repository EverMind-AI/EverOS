"""Private helpers shared across the search e2e tests.

* :func:`pick_query_seeds` — scans the session corpus's
  ``episodes/`` md files and returns a list of ``(owner_id, subject)``
  tuples to use as deterministic search queries. Bootstrapping queries
  off the corpus's own extraction output gives us a closed-loop
  correctness signal — what was written should be findable.

* :func:`assert_recall` — the canonical "this search returned at
  least one sensible hit for ``owner``" assertion bundle, so the
  assertion logic is in one place.

* :func:`flatten_hits` — collapses ``SearchData``'s episode and profile
  arrays into one ``(owner_id, score, text)`` tuple list.

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
    """Sample ``(owner_id, subject)`` tuples from the episode md files.

    Walks ``users/<owner>/episodes/episode-*.md`` and parses the
    ``### Subject`` sections inside each daily-log entry. Returns
    deterministic seeds (the explicit ``sorted`` calls) so a flaky test
    surfaces a real regression, not query-rotation variance.

    Raises:
        AssertionError: if no episode was extracted — that's a fixture
            failure, not a test failure, and should fail loudly.
    """
    seeds: list[tuple[str, str]] = []
    users_dir = memory_root / "default_app" / "default_project" / "users"
    if not users_dir.is_dir():
        raise AssertionError(f"expected {users_dir} to exist after ingest")

    for owner_dir in sorted(users_dir.iterdir()):
        episodes_dir = owner_dir / "episodes"
        if not episodes_dir.is_dir():
            continue
        for md in sorted(episodes_dir.rglob("*.md")):
            for subject in _extract_sections(md, "### Subject"):
                seeds.append((owner_dir.name, subject))
                if len(seeds) >= limit:
                    return seeds
    if not seeds:
        raise AssertionError(
            f"no episode md entries under {users_dir} — pipeline did "
            "not produce any episodes; cannot bootstrap search queries"
        )
    return seeds


def _extract_sections(md: Path, heading: str) -> list[str]:
    """Return every ``heading`` section body in a daily-log md file.

    Daily-log entries are ``## <entry-id>`` blocks; the labelled body
    sections inside an entry are h3 (``### Subject``, ``### Summary``,
    …). We scan linearly for ``heading`` and collect lines until the
    next heading at any level or the end-of-entry marker.
    """
    body = md.read_text(encoding="utf-8")
    sections: list[str] = []
    inside = False
    buf: list[str] = []
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(heading):
            if inside:
                sections.append("\n".join(buf).strip())
            inside = True
            buf = []
            continue
        # Any subsequent heading or entry-end marker closes the section.
        if inside and (stripped.startswith("#") or stripped.startswith("<!-- /entry")):
            sections.append("\n".join(buf).strip())
            inside = False
            buf = []
            continue
        if inside:
            buf.append(line)
    if inside:
        sections.append("\n".join(buf).strip())
    return [s for s in sections if s]


# ── Response flattening + assertions ────────────────────────────────────


def flatten_hits(data: dict[str, Any]) -> list[tuple[str | None, float, str]]:
    """Collapse ``SearchData``'s episodes and profiles into ``(owner, score, text)``.

    Stable shape so the recall / partition tests don't have to branch.
    ``owner`` may be ``None`` for profile hits where the owner is implicit.
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
    return out


async def assert_recall(
    client: httpx.AsyncClient,
    *,
    owner_id: str,
    query: str,
    min_score: float = 0.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Hit ``/search`` (multi-round) and lock the four standard recall invariants.

    1. **Status** 200 — the route compiled.
    2. **Existence** — ``total >= 1`` across the episodes and profiles.
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
            "method": "llm_multiround",
            "top_k": top_k,
        },
        timeout=300.0,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    hits = flatten_hits(body["data"])
    assert hits, f"no hits for owner={owner_id} query={query!r} — recall is broken"
    for hit_owner, _score, _text in hits:
        if hit_owner is not None:
            assert hit_owner == owner_id, (
                f"partition leak: got owner={hit_owner!r} when querying {owner_id!r}"
            )
    top_score = max(score for _o, score, _t in hits)
    assert top_score >= min_score, f"top hit score {top_score:.3f} < min {min_score}"
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
