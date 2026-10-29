"""``SearchManager._validate_components`` — capability gate at request entry.

Pins the fail-fast contract: a search ``method`` whose recall/rerank path
needs a provider that is not configured must raise
:class:`ProviderNotConfiguredError` (-> HTTP 422) *before* any recall runs,
with the right ``provider`` / ``feature`` / ``alternative_hint`` payload.

The gate reads the providers this manager was built with: a ``None``
provider fails the guard, so dispatch never reaches an
``AttributeError('NoneType')`` deep in recall.

Agent HYBRID has two rerank lanes selected by ``enable_llm_rerank``: the
LLM lane (``True``) reranks via the LLM and needs no rerank provider; the
cross-encoder lane (``False``, default) does.
"""

from __future__ import annotations

import pytest

from everos.core.errors import ProviderNotConfiguredError
from everos.memory.search.dto import SearchMethod, SearchRequest
from everos.memory.search.manager import SearchManager


class _StubLLM:
    """Marker instance -- ``_validate_components`` only checks it for None-ness."""


class _StubEmbedding:
    """Marker instance -- guard only checks it for None-ness."""


class _StubReranker:
    """Marker instance -- guard only checks it for None-ness."""


def _build_manager(
    *,
    llm_present: bool = True,
    embedding_present: bool = False,
    reranker_present: bool = False,
) -> SearchManager:
    """A manager whose recallers are never touched -- ``_validate_components``
    runs before any recall dispatch, so only ``llm_client`` / ``embedding``
    / ``reranker`` presence matters here.
    """
    return SearchManager(
        episode_recaller=None,  # type: ignore[arg-type]
        atomic_fact_recaller=None,  # type: ignore[arg-type]
        agent_case_recaller=None,  # type: ignore[arg-type]
        agent_skill_recaller=None,  # type: ignore[arg-type]
        profile_recaller=None,  # type: ignore[arg-type]
        session_buffer=None,  # type: ignore[arg-type]
        cluster_store=None,  # type: ignore[arg-type]
        embedding=_StubEmbedding() if embedding_present else None,  # type: ignore[arg-type]
        reranker=_StubReranker() if reranker_present else None,  # type: ignore[arg-type]
        llm_client=_StubLLM() if llm_present else None,
    )


@pytest.fixture
def manager() -> SearchManager:
    """LLM is always wired -- it's a Tier-1 hard requirement enforced at
    server startup, so every case here assumes it's present and only
    varies embedding / rerank availability.

    ``embedding`` and ``reranker`` default to ``None`` here so tests
    exercising the "provider unavailable" branch don't need to opt out;
    tests that need to reach past the embedding guard build a manager
    with ``embedding_present=True`` (and ``reranker_present=True`` for
    rerank-dependent branches) via :func:`_build_manager` directly.
    """
    return _build_manager()


# ── VECTOR / user HYBRID: embed is the only prerequisite ──────────────────


def test_vector_without_embed_raises_embedding_error(manager: SearchManager) -> None:
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.VECTOR)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "vector"


def test_hybrid_without_embed_raises_embedding_error(manager: SearchManager) -> None:
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.HYBRID)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "user_hybrid"


# ── AGENTIC: embed + rerank ─────────────────────────────────────────────


def test_agentic_without_embed_raises_embedding_error(manager: SearchManager) -> None:
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.AGENTIC)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "agentic_search"


def test_agentic_without_rerank_raises_rerank_error() -> None:
    """Embedding injected -- the embedding guard passes and the rerank
    branch is what fails."""
    manager = _build_manager(embedding_present=True)
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.AGENTIC)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "rerank"
    assert excinfo.value.feature == "agentic_search"


# ── Agent HYBRID: dual rerank lane ─────────────────────────────────────


def test_agent_hybrid_default_lane_without_rerank_raises_with_hint() -> None:
    """agent HYBRID + enable_llm_rerank=False (default) + no rerank -> 422 with hint."""
    manager = _build_manager(embedding_present=True)
    req = SearchRequest(
        agent_id="a1", query="q", method=SearchMethod.HYBRID, enable_llm_rerank=False
    )
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "rerank"
    assert excinfo.value.feature == "agent_hybrid"
    assert excinfo.value.alternative_hint is not None
    assert "enable_llm_rerank" in excinfo.value.alternative_hint


def test_agent_hybrid_llm_lane_without_rerank_passes() -> None:
    """agent HYBRID + enable_llm_rerank=True -> uses LLM, no rerank needed."""
    manager = _build_manager(embedding_present=True)
    req = SearchRequest(
        agent_id="a1", query="q", method=SearchMethod.HYBRID, enable_llm_rerank=True
    )
    manager._validate_components(req)  # must not raise


def test_agent_hybrid_llm_lane_without_llm_raises() -> None:
    """LLM absence is defensive-only (a running server can't start without
    one) but the gate must still refuse rather than dispatch into a
    rerank call with no LLM client wired. Uniform with the embedding /
    rerank branches: 422 via :class:`ProviderNotConfiguredError`."""
    manager = _build_manager(llm_present=False, embedding_present=True)
    req = SearchRequest(
        agent_id="a1", query="q", method=SearchMethod.HYBRID, enable_llm_rerank=True
    )
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "llm"
    assert excinfo.value.feature == "agent_hybrid"
    assert excinfo.value.alternative_hint is not None
    assert "enable_llm_rerank" in excinfo.value.alternative_hint


# ── LLM branches: hybrid+enable_llm_rerank and AGENTIC both 422 ─────────


def test_hybrid_llm_lane_missing_client_raises_provider_not_configured() -> None:
    """User HYBRID + ``enable_llm_rerank=True`` routes through the LLM
    rerank lane; a manager built without an LLM client must fail with a
    422 that points at ``[llm]`` and offers the cross-encoder fallback."""
    manager = _build_manager(llm_present=False, embedding_present=True)
    req = SearchRequest(
        user_id="u1", query="q", method=SearchMethod.HYBRID, enable_llm_rerank=True
    )
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "llm"
    assert excinfo.value.feature == "user_hybrid"
    assert "[llm]" in str(excinfo.value)
    assert excinfo.value.alternative_hint is not None
    assert "enable_llm_rerank" in excinfo.value.alternative_hint


def test_agentic_missing_llm_client_raises_provider_not_configured() -> None:
    """AGENTIC needs an LLM to drive the agentic loop; without one the
    gate must fail with 422 and point at ``[llm]``."""
    manager = _build_manager(
        llm_present=False, embedding_present=True, reranker_present=True
    )
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.AGENTIC)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "llm"
    assert excinfo.value.feature == "agentic_search"
    assert "[llm]" in str(excinfo.value)


def test_gate_passes_when_every_provider_is_injected() -> None:
    """Happy path: every method passes when its providers were injected."""
    manager = _build_manager(embedding_present=True, reranker_present=True)
    for method in (
        SearchMethod.KEYWORD,
        SearchMethod.VECTOR,
        SearchMethod.HYBRID,
        SearchMethod.AGENTIC,
    ):
        req = SearchRequest(user_id="u1", query="q", method=method)
        manager._validate_components(req)  # no raise


def test_keyword_never_requires_embed_or_rerank(manager: SearchManager) -> None:
    """KEYWORD has no provider prerequisites -- passes with both injected
    providers ``None``."""
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.KEYWORD)
    manager._validate_components(req)  # no raise
