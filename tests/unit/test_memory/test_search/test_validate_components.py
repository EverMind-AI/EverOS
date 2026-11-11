"""``SearchManager._validate_components`` — capability gate at request entry.

Pins the fail-fast contract: multi-round retrieval embeds every sub-query,
so a manager built without an embedding provider must raise
:class:`ProviderNotConfiguredError` (-> HTTP 422) *before* any recall runs,
with the right ``provider`` / ``feature`` payload.

The gate reads the provider this manager was built with: a ``None``
provider fails the guard, so dispatch never reaches an
``AttributeError('NoneType')`` deep in recall. The LLM is not gated: the
manager cannot be built without one, and no rerank provider is needed --
multi-round retrieval has no cross-encoder stage.
"""

from __future__ import annotations

import pytest

from everos.core.errors import ProviderNotConfiguredError
from everos.memory.access import AccessPolicy
from everos.memory.search.dto import SearchRequest
from everos.memory.search.manager import SearchManager
from tests.fakes import InMemoryGroupStore, static_trace_path, static_tuning


class _StubLLM:
    """Marker instance -- ``_validate_components`` never reads it."""


class _StubEmbedding:
    """Marker instance -- guard only checks it for None-ness."""


def _build_manager(*, embedding_present: bool) -> SearchManager:
    """A manager whose recallers are never touched -- ``_validate_components``
    runs before any recall dispatch, so only ``embedding`` presence matters.
    """
    return SearchManager(
        access=AccessPolicy(InMemoryGroupStore()),
        episode_recaller=None,  # type: ignore[arg-type]
        profile_recaller=None,  # type: ignore[arg-type]
        session_buffer=None,  # type: ignore[arg-type]
        embedding=_StubEmbedding() if embedding_present else None,  # type: ignore[arg-type]
        decider_client=_StubLLM(),  # type: ignore[arg-type]
        tuning_source=static_tuning(),
        trace_path_source=static_trace_path(),
    )


def test_no_embedding_is_refused_before_recall() -> None:
    manager = _build_manager(embedding_present=False)
    with pytest.raises(ProviderNotConfiguredError) as exc:
        manager._validate_components()
    assert exc.value.provider == "embedding"
    assert exc.value.feature == "llm_multiround"


def test_an_embedding_provider_passes_without_a_reranker() -> None:
    _build_manager(embedding_present=True)._validate_components()


async def test_search_refuses_before_any_recall_without_embedding() -> None:
    """The guard fires from the public entry: the ``None`` recallers would
    raise ``AttributeError`` if dispatch got past it."""
    manager = _build_manager(embedding_present=False)
    with pytest.raises(ProviderNotConfiguredError):
        await manager.search(SearchRequest(user_id="alice", query="hello"))
