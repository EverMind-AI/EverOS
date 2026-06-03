"""memory.search — read path: hybrid retrieval over LanceDB.

This subpackage owns the recall + adapter layer for ``POST
/api/v1/memory/search``. All fusion / rerank / agentic algorithms are
delegated to :mod:`everalgo.rank`; this layer is responsible for:

* compiling the Filters DSL into a LanceDB ``where`` string,
* per-kind sparse (BM25 over ``*_tokens``) and dense (vector ANN) recall,
* shape-translating ``everalgo.rank.RankOutput`` into the public DTOs.

Cascade writes drive the underlying LanceDB rows; this package never
writes.

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
