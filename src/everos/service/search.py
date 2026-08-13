"""Search use case — lazy singleton wiring for the public search endpoint.

Mirrors the lazy-build pattern in :mod:`everos.service.memorize`: the
manager and all its dependencies are constructed on first call so that
the FastAPI module-level import order doesn't conflict with the
lifespan that brings up LanceDB / settings.

Component policy (matches :class:`SearchManager` guards):

* Embedding / rerank / LLM clients are **optional at boot**; the manager
  is built lazily and only the methods that need them fail (with a clear
  message) when the corresponding section of settings is empty.
* ``KEYWORD`` searches therefore work without any of the three clients,
  which makes the endpoint usable in a freshly-installed dev setup.
* All three providers are pulled from their process-wide capability /
  singleton accessors (:func:`get_embedding_capability`,
  :func:`get_rerank_capability`, :func:`get_llm_client`); this module
  never keeps parallel singletons of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.component.embedding import get_embedding_capability
from everos.component.llm import LLMNotConfiguredError, get_llm_client
from everos.component.rerank import get_rerank_capability
from everos.component.tokenizer import build_tokenizer
from everos.core.observability.logging import get_logger
from everos.memory.search import SearchRequest, SearchResponse
from everos.memory.search.manager import SearchManager
from everos.memory.search.recall import (
    AgentCaseRecaller,
    AgentSkillRecaller,
    AtomicFactRecaller,
    EpisodeRecaller,
    ProfileRecaller,
    RecallerDeps,
)

if TYPE_CHECKING:
    from everos.component.llm import LLMClient

logger = get_logger(__name__)

# Lazy singleton — the manager itself; every provider it needs comes
# from its own process-wide accessor (see module docstring).
_manager: SearchManager | None = None


def _get_llm_client() -> LLMClient | None:
    """Return the process-wide LLM client, or ``None`` when unset.

    Delegates to :func:`everos.component.llm.get_llm_client` (which
    caches its own singleton, validates ``api_key`` / ``base_url``, and
    already wraps the client with ``UsageRecordingClient`` when
    observability is enabled — see ``component/llm/client.py``) instead
    of maintaining a parallel module-level singleton here. LLM is
    optional for search — ``KEYWORD`` works without it — so
    :class:`LLMNotConfiguredError` is swallowed into ``None`` and the
    manager surfaces a clear error only when a method that actually
    needs the LLM is invoked.
    """
    try:
        return get_llm_client()
    except LLMNotConfiguredError:
        logger.warning(
            "llm_not_configured",
            hint="set [llm] api_key / base_url to enable hybrid / agentic search",
        )
        return None


def _get_manager() -> SearchManager:
    global _manager
    if _manager is None:
        deps = RecallerDeps(tokenizer=build_tokenizer())
        _manager = SearchManager(
            episode_recaller=EpisodeRecaller(deps),
            atomic_fact_recaller=AtomicFactRecaller(deps),
            agent_case_recaller=AgentCaseRecaller(deps),
            agent_skill_recaller=AgentSkillRecaller(deps),
            profile_recaller=ProfileRecaller(),
            embedding=get_embedding_capability().provider,
            reranker=get_rerank_capability().provider,
            llm_client=_get_llm_client(),
        )
    return _manager


async def search(req: SearchRequest) -> SearchResponse:
    """Dispatch one search request through the lazily-built manager."""
    return await _get_manager().search(req)
