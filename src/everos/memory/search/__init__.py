"""memory.search — read path: multi-round retrieval over the derived indexes.

This subpackage owns ``POST /api/v2/memory/search``: it compiles the
Filters DSL into a backend-neutral :class:`~everos.memory.ports.Filter`,
recalls episodes and pictures through the ``*Index`` ports, lets the
multi-round decider select the core, and shapes the public DTOs. Multi-round
retrieval (``llm_multiround``) is the only search method. It never writes;
the cascade maintains the indexes.

Where everalgo's ``Candidate`` type may appear -- three places, nowhere else:

* ``recall/`` -- the inbound mapper: index ``SearchHit`` -> ``Candidate``,
  with the stored domain records returned beside the candidates
  (``RecallResult.stored_by_id``), never inside ``Candidate.metadata``.
* ``shaper.py`` -- the outbound mapper: ranked ``Candidate`` + stored
  records -> response DTOs.
* ``llm_multiround.py`` -- the retrieval loop (iterative per-sub-query blocks
  with an LLM core selector), hosted here pending upstreaming to everalgo.
  ``callbacks.py`` keeps the cross-encoder ``RerankFn`` factory knowledge
  search uses.

``manager.py`` orchestrates: it resolves the read scope, runs the loop,
attaches profiles and referenced pictures, and neither constructs nor
imports an everalgo candidate type.

External usage::

    from everos.memory.search import (
        ALLOWED_FIELDS,
        RESERVED_FIELDS,
        FilterError,
        FilterNode,
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
from .dto import SearchData as SearchData
from .dto import SearchEpisodeItem as SearchEpisodeItem
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
from .shaper import group_source as group_source

__all__ = [
    "ALLOWED_FIELDS",
    "RESERVED_FIELDS",
    "FilterError",
    "FilterNode",
    "SearchData",
    "SearchEpisodeItem",
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
]
