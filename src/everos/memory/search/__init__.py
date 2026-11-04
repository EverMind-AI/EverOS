"""memory.search — read path: hybrid retrieval over the derived indexes.

This subpackage owns ``POST /api/v2/memory/search``: it compiles the
Filters DSL into a backend-neutral :class:`~everos.memory.ports.Filter`,
recalls per kind through the ``*Index`` ports, ranks, and shapes the
public DTOs. It never writes; the cascade maintains the indexes.

Where everos.algo's candidate types (``Candidate`` / ``RankInput`` /
``ScoredItem``) meet the host:

* ``recall/`` -- the inbound mapper: index ``SearchHit`` -> ``Candidate``,
  with the stored domain records returned beside the candidates
  (``RecallResult.stored_by_id``), never inside ``Candidate.metadata``.
* ``shaper.py`` -- the outbound mapper: ranked ``Candidate`` /
  ``ScoredItem`` + stored records -> response DTOs.
* ``everos.algo.rank`` owns stateless heap expansion and multi-round evidence
  selection. This package supplies candidates and per-request state, then
  shapes results and persists traces.
* Retrieval orchestration remains here:
  ``llm_multiround.py`` (iterative per-sub-query blocks with an LLM core
  selector), ``skill_hybrid.py`` (RRF + skill cross-encoder lane),
  ``callbacks.py`` (kind-shaped cross-encoder ``RerankFn`` factories),
  ``agentic.py`` and ``agentic_agent.py`` (compositions of the everos.algo
  maxsim / hybrid / cluster / agentic facades). Further stateless logic can
  move into ``everos.algo`` without moving stores or runtime ownership.

``manager.py`` orchestrates: it picks the route per kind, passes recall
results to the algorithms and the shaper, and neither constructs nor
imports an everos.algo candidate type.

External usage::

    from everos.memory.search import (
        ALLOWED_FIELDS,
        RESERVED_FIELDS,
        FilterError,
        FilterNode,
        SearchAgentCaseItem,
        SearchAgentSkillItem,
        SearchAtomicFactItem,
        SearchData,
        SearchEpisodeItem,
        SearchMethod,
        SearchProfileItem,
        SearchRequest,
        SearchResponse,
        compile_filters,
        compile_predicate,
    )

The Filters DSL primitives (``ALLOWED_FIELDS`` / ``RESERVED_FIELDS`` /
``compile_predicate``) are re-exported so :mod:`memory.get` can build
its flat-DSL variant without forking the field allow-list.
"""

from .dto import FilterNode as FilterNode
from .dto import SearchAgentCaseItem as SearchAgentCaseItem
from .dto import SearchAgentSkillItem as SearchAgentSkillItem
from .dto import SearchAtomicFactItem as SearchAtomicFactItem
from .dto import SearchData as SearchData
from .dto import SearchEpisodeItem as SearchEpisodeItem
from .dto import SearchMethod as SearchMethod
from .dto import SearchProfileItem as SearchProfileItem
from .dto import SearchRequest as SearchRequest
from .dto import SearchResponse as SearchResponse
from .filters import ALLOWED_FIELDS as ALLOWED_FIELDS
from .filters import RESERVED_FIELDS as RESERVED_FIELDS
from .filters import FilterError as FilterError
from .filters import compile_filters as compile_filters
from .filters import compile_predicate as compile_predicate

__all__ = [
    "ALLOWED_FIELDS",
    "RESERVED_FIELDS",
    "FilterError",
    "FilterNode",
    "SearchAgentCaseItem",
    "SearchAgentSkillItem",
    "SearchAtomicFactItem",
    "SearchData",
    "SearchEpisodeItem",
    "SearchMethod",
    "SearchProfileItem",
    "SearchRequest",
    "SearchResponse",
    "compile_filters",
    "compile_predicate",
]
