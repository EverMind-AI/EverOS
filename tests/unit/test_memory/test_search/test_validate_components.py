"""``SearchManager._validate_components`` — capability gate at request entry.

Pins the fail-fast contract: a search ``method`` whose recall/rerank path
needs a provider that is not configured must raise
:class:`ProviderNotConfiguredError` (-> HTTP 422) *before* any recall runs,
with the right ``provider`` / ``feature`` / ``alternative_hint`` payload.

Belt-and-suspenders: the gate reads BOTH sources of truth -- the
process-wide capability singleton accessors
(``get_embedding_capability`` / ``get_rerank_capability``) AND this
manager's own injected providers. Either being unavailable / ``None``
fails the guard, so dispatch never reaches an
``AttributeError('NoneType')`` deep in recall.

Agent HYBRID has two rerank lanes selected by ``enable_llm_rerank``: the
LLM lane (``True``) reranks via the LLM and needs no rerank provider; the
cross-encoder lane (``False``, default) does.
"""

from __future__ import annotations

import pytest

import everos.component.embedding.accessor as embedding_accessor
import everos.component.rerank.accessor as rerank_accessor
from everos.component.embedding import EmbeddingCapability
from everos.component.rerank import RerankCapability
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

    Under belt-and-suspenders the guard cross-checks accessor availability
    against the manager's own injected providers, so tests that want to
    reach a downstream branch must line up both sides.
    """
    return SearchManager(
        episode_recaller=None,  # type: ignore[arg-type]
        atomic_fact_recaller=None,  # type: ignore[arg-type]
        agent_case_recaller=None,  # type: ignore[arg-type]
        agent_skill_recaller=None,  # type: ignore[arg-type]
        profile_recaller=None,  # type: ignore[arg-type]
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


@pytest.fixture(autouse=True)
def _capabilities_unavailable_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default both capability singletons to unavailable; tests opt into
    availability via the ``embed_available`` / ``rerank_available`` fixtures."""
    monkeypatch.setattr(
        embedding_accessor, "_capability", EmbeddingCapability(provider=None)
    )
    monkeypatch.setattr(rerank_accessor, "_capability", RerankCapability(provider=None))


@pytest.fixture
def embed_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embedding_accessor, "_capability", EmbeddingCapability(provider=object())
    )


@pytest.fixture
def rerank_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rerank_accessor, "_capability", RerankCapability(provider=object())
    )


@pytest.fixture
def rerank_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit alias for the (already-default) unavailable state -- kept
    for call-site readability when paired with ``embed_available``."""
    monkeypatch.setattr(rerank_accessor, "_capability", RerankCapability(provider=None))


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


def test_agentic_without_embed_raises_embedding_error(
    manager: SearchManager, rerank_available: None
) -> None:
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.AGENTIC)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "agentic_search"


def test_agentic_without_rerank_raises_rerank_error(embed_available: None) -> None:
    """Embed accessor + injected embedding both present -- the embedding
    guard passes and the rerank branch is what fails."""
    manager = _build_manager(embedding_present=True)
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.AGENTIC)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "rerank"
    assert excinfo.value.feature == "agentic_search"


# ── Agent HYBRID: dual rerank lane ─────────────────────────────────────


def test_agent_hybrid_default_lane_without_rerank_raises_with_hint(
    embed_available: None, rerank_unavailable: None
) -> None:
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


def test_agent_hybrid_llm_lane_without_rerank_passes(
    embed_available: None, rerank_unavailable: None
) -> None:
    """agent HYBRID + enable_llm_rerank=True -> uses LLM, no rerank needed."""
    manager = _build_manager(embedding_present=True)
    req = SearchRequest(
        agent_id="a1", query="q", method=SearchMethod.HYBRID, enable_llm_rerank=True
    )
    manager._validate_components(req)  # must not raise


def test_agent_hybrid_llm_lane_without_llm_raises(embed_available: None) -> None:
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


def test_hybrid_llm_lane_missing_client_raises_provider_not_configured(
    embed_available: None,
) -> None:
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


def test_agentic_missing_llm_client_raises_provider_not_configured(
    embed_available: None, rerank_available: None
) -> None:
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


# ── Belt-and-suspenders: injected-None case ────────────────────────────


def test_gate_fails_when_accessor_available_but_self_embedding_missing(
    manager: SearchManager, embed_available: None
) -> None:
    """Accessor says embedding is configured, but this manager was built
    with ``embedding=None`` (e.g. a Tier-1 KEYWORD-only path sharing the
    same SearchManager). The guard must fail cleanly with 422 rather than
    let dispatch reach ``self._embedding.embed(query)`` and crash on
    ``AttributeError('NoneType')``."""
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.VECTOR)
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "vector"


def test_gate_fails_when_accessor_available_but_self_reranker_missing(
    embed_available: None, rerank_available: None
) -> None:
    """Accessor says rerank is configured, but this manager was built
    with ``reranker=None``. Agent HYBRID cross-encoder lane must still
    refuse rather than dispatch into a rerank call with no provider."""
    manager = _build_manager(embedding_present=True)
    req = SearchRequest(
        agent_id="a1", query="q", method=SearchMethod.HYBRID, enable_llm_rerank=False
    )
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        manager._validate_components(req)
    assert excinfo.value.provider == "rerank"
    assert excinfo.value.feature == "agent_hybrid"


def test_gate_passes_when_both_accessor_and_injected_present(
    embed_available: None, rerank_available: None
) -> None:
    """Happy path: both sides of the AND agree the provider exists, for
    every method that requires one."""
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
    """KEYWORD has no provider prerequisites -- passes even with both
    capabilities unavailable (the autouse default state) and both
    injected providers ``None``."""
    req = SearchRequest(user_id="u1", query="q", method=SearchMethod.KEYWORD)
    manager._validate_components(req)  # no raise
