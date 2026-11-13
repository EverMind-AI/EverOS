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

``manager.py`` orchestrates: it resolves the read scope, runs the loop,
attaches profiles and referenced pictures, and neither constructs nor
imports an everalgo candidate type. ``forgetting.py`` is the forgetting
curve's arithmetic (retention, a version's state, reinforcement), off by
default; the loop reweighs its candidates with it. ``past_triggers.py`` reads
a question's past-time wording (``[search] past_triggers``, off by default),
which sends a request that leaves ``include_past`` unset to the past.

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

from .audience import with_audience as with_audience
from .decider_health import DeciderHealth as DeciderHealth
from .decider_health import DeciderStatus as DeciderStatus
from .dto import NO_AGENT_MEMORY as NO_AGENT_MEMORY
from .dto import AudienceItem as AudienceItem
from .dto import FilterNode as FilterNode
from .dto import GroupProfileEntry as GroupProfileEntry
from .dto import ProfileSourceItem as ProfileSourceItem
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
from .forgetting import Curve as Curve
from .forgetting import reinforcements as reinforcements
from .forgetting import strengthen as strengthen
from .group_profile import item_versions as item_versions
from .group_profile import shape_profile_item as shape_profile_item
from .group_profile import visible_profile as visible_profile
from .group_profile import visible_profile_items as visible_profile_items
from .recall import ProfileView as ProfileView
from .recall import fetch_member_profile as fetch_member_profile
from .recall import fetch_user_profile as fetch_user_profile
from .shaper import group_source as group_source

__all__ = [
    "ALLOWED_FIELDS",
    "NO_AGENT_MEMORY",
    "RESERVED_FIELDS",
    "AudienceItem",
    "Curve",
    "DeciderHealth",
    "DeciderStatus",
    "FilterError",
    "FilterNode",
    "GroupProfileEntry",
    "ProfileSourceItem",
    "ProfileView",
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
    "fetch_member_profile",
    "fetch_user_profile",
    "group_source",
    "item_versions",
    "reinforcements",
    "shape_profile_item",
    "strengthen",
    "visible_profile",
    "visible_profile_items",
    "with_audience",
]
