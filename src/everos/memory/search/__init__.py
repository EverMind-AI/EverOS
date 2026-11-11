"""memory.search — read path: hybrid retrieval over the derived indexes.

This subpackage owns ``POST /api/v2/memory/search``: it compiles the
Filters DSL into a backend-neutral :class:`~everos.memory.ports.Filter`,
recalls per kind through the ``*Index`` ports, ranks, and shapes the
public DTOs. It never writes; the cascade maintains the indexes.

Where everalgo's candidate types (``Candidate`` / ``RankInput`` /
``ScoredItem``) may appear -- three groups, nowhere else:

* ``recall/`` -- the inbound mapper: index ``SearchHit`` -> ``Candidate``,
  with the stored domain records returned beside the candidates
  (``RecallResult.stored_by_id``), never inside ``Candidate.metadata``.
* ``shaper.py`` -- the outbound mapper: ranked ``Candidate`` /
  ``ScoredItem`` + stored records -> response DTOs.
* Six rank algorithms still hosted here, pending upstreaming to
  everalgo-rank (they speak ``Candidate`` natively and move as a whole):
  ``hierarchy.py`` (heap-expand episode/fact competition),
  ``llm_multiround.py`` (iterative per-sub-query blocks with an LLM core
  selector), ``skill_hybrid.py`` (RRF + skill cross-encoder lane),
  ``callbacks.py`` (kind-shaped cross-encoder ``RerankFn`` factories),
  ``agentic.py`` and ``agentic_agent.py`` (compositions of the everalgo
  maxsim / hybrid / cluster / agentic facades). Once everalgo ships them,
  each file is replaced by a call to the upstream entry.

``manager.py`` orchestrates: it picks the route per kind, passes recall
results to the algorithms and the shaper, and neither constructs nor
imports an everalgo candidate type.

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

from .audience import with_audience as with_audience
from .dto import AudienceItem as AudienceItem
from .dto import FilterNode as FilterNode
from .dto import GroupProfileEntry as GroupProfileEntry
from .dto import ProfileSourceItem as ProfileSourceItem
from .dto import SearchAgentCaseItem as SearchAgentCaseItem
from .dto import SearchAgentSkillItem as SearchAgentSkillItem
from .dto import SearchAtomicFactItem as SearchAtomicFactItem
from .dto import SearchData as SearchData
from .dto import SearchEpisodeItem as SearchEpisodeItem
from .dto import SearchGroupProfile as SearchGroupProfile
from .dto import SearchMethod as SearchMethod
from .dto import SearchProfileItem as SearchProfileItem
from .dto import SearchRequest as SearchRequest
from .dto import SearchResponse as SearchResponse
from .dto import SearchScope as SearchScope
from .dto import SearchSource as SearchSource
from .dto import SearchWindow as SearchWindow
from .filters import ALLOWED_FIELDS as ALLOWED_FIELDS
from .filters import RESERVED_FIELDS as RESERVED_FIELDS
from .filters import FilterError as FilterError
from .filters import compile_filters as compile_filters
from .filters import compile_predicate as compile_predicate
from .filters import compile_scoped as compile_scoped
from .group_profile import item_versions as item_versions
from .group_profile import shape_profile_item as shape_profile_item
from .group_profile import visible_profile as visible_profile
from .group_profile import visible_profile_items as visible_profile_items
from .shaper import group_source as group_source

__all__ = [
    "ALLOWED_FIELDS",
    "RESERVED_FIELDS",
    "AudienceItem",
    "FilterError",
    "FilterNode",
    "GroupProfileEntry",
    "ProfileSourceItem",
    "SearchAgentCaseItem",
    "SearchAgentSkillItem",
    "SearchAtomicFactItem",
    "SearchData",
    "SearchEpisodeItem",
    "SearchGroupProfile",
    "SearchMethod",
    "SearchProfileItem",
    "SearchRequest",
    "SearchResponse",
    "SearchScope",
    "SearchSource",
    "SearchWindow",
    "compile_filters",
    "compile_predicate",
    "compile_scoped",
    "group_source",
    "item_versions",
    "shape_profile_item",
    "visible_profile",
    "visible_profile_items",
    "with_audience",
]
