"""memory.get — read path: paginated listing over LanceDB.

This subpackage owns the dispatch + shape layer for ``POST
/api/v1/memory/get``. Unlike :mod:`memory.search`, /get does no
ranking — it is a pure offset/limit + scalar-filter listing,
partitioned by ``(owner_type, memory_type)``.

Filters reuse :class:`everos.memory.search.FilterNode` and the
shared :func:`compile_filters` path — same DSL surface, same compile
output, ``AND`` / ``OR`` combinators allowed.

External usage::

    from everos.memory.get import (
        GetAgentCaseItem,
        GetAgentSkillItem,
        GetData,
        GetEpisodeItem,
        GetManager,
        GetMemoryType,
        GetProfileItem,
        GetRequest,
        GetResponse,
        compile_filters_for_get,
    )
"""

from .dto import GetAgentCaseItem as GetAgentCaseItem
from .dto import GetAgentSkillItem as GetAgentSkillItem
from .dto import GetData as GetData
from .dto import GetEpisodeItem as GetEpisodeItem
from .dto import GetMemoryType as GetMemoryType
from .dto import GetProfileItem as GetProfileItem
from .dto import GetRequest as GetRequest
from .dto import GetResponse as GetResponse
from .filters_adapter import compile_filters_for_get as compile_filters_for_get
from .manager import GetManager as GetManager

__all__ = [
    "GetAgentCaseItem",
    "GetAgentSkillItem",
    "GetData",
    "GetEpisodeItem",
    "GetManager",
    "GetMemoryType",
    "GetProfileItem",
    "GetRequest",
    "GetResponse",
    "compile_filters_for_get",
]
