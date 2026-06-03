"""Search use case — lazy singleton wiring for the public search endpoint.

Mirrors the lazy-build pattern in :mod:`everos.service.memorize`: the
manager and all its dependencies are constructed on first call so that
the FastAPI module-level import order doesn't conflict with the
lifespan that brings up LanceDB / settings.

Component policy (matches :class:`SearchManager` guards):

* Embedding / rerank / LLM clients are **optional at boot**; they are
  built lazily, and only the methods that need them fail (with a clear
  message) when the corresponding section of settings is empty.
* ``KEYWORD`` searches therefore work without any of the three clients,
  which makes the endpoint usable in a freshly-installed dev setup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from everos.component.embedding import EmbeddingProvider
    from everos.component.llm import LLMClient
    from everos.component.rerank import RerankProvider

logger = get_logger(__name__)

# Lazy singletons ────────────────────────────────────────────────────────

_manager: SearchManager | None = None
_embedding: EmbeddingProvider | None = None
_reranker: RerankProvider | None = None
_llm_client: LLMClient | None = None
_embedding_resolved = False
_rerank_resolved = False
_llm_resolved = False


def _get_embedding() -> EmbeddingProvider | None:
    """Build the embedding client on first call. ``None`` when not configured."""
    global _embedding, _embedding_resolved
    if _embedding_resolved:
        return _embedding

    from everos.component.embedding import build_embedding_provider
    from everos.config import load_settings

    cfg = load_settings().embedding
    if not cfg.model or cfg.api_key is None:
        logger.warning(
            "embedding_not_configured",
            hint="set [embedding] model / api_key to enable vector / hybrid search",
        )
        _embedding = None
    else:
        _embedding = build_embedding_provider(cfg)
        logger.info("search_embedding_built", model=cfg.model)
    _embedding_resolved = True
    return _embedding


def _get_reranker() -> RerankProvider | None:
    """Build the rerank client on first call. ``None`` when not configured."""
    global _reranker, _rerank_resolved
    if _rerank_resolved:
        return _reranker

    from everos.component.rerank import build_rerank_provider
    from everos.config import load_settings

    cfg = load_settings().rerank
    if not cfg.model or not cfg.base_url:
        logger.warning(
            "rerank_not_configured",
            hint="set [rerank] model / base_url to enable agentic search",
        )
        _reranker = None
    else:
        _reranker = build_rerank_provider(cfg)
        logger.info("search_rerank_built", model=cfg.model, provider=cfg.provider)
    _rerank_resolved = True
    return _reranker


def _get_llm_client() -> LLMClient | None:
    """Lazily build the LLM client from settings (shared with memorize)."""
    global _llm_client, _llm_resolved
    if _llm_resolved:
        return _llm_client

    from everos.component.llm import build_llm_provider
    from everos.config import load_settings

    cfg = load_settings().llm
    if cfg.api_key is None or not cfg.base_url:
        logger.warning(
            "llm_not_configured",
            hint="set [llm] api_key / base_url to enable hybrid / agentic search",
        )
        _llm_client = None
    else:
        _llm_client = build_llm_provider(cfg)
        logger.info("search_llm_built", model=cfg.model)
    _llm_resolved = True
    return _llm_client


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
            embedding=_get_embedding(),
            reranker=_get_reranker(),
            llm_client=_get_llm_client(),
        )
    return _manager


async def search(req: SearchRequest) -> SearchResponse:
    """Dispatch one search request through the lazily-built manager."""
    return await _get_manager().search(req)
