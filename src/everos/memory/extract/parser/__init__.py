"""memory.extract.parser — multimodal ContentItem parsing via everalgo.parser.

External usage:
    from everos.memory.extract.parser import (
        enrich_content_items,
        has_unparsed_multimodal,
        multimodal_available,
        require_multimodal,
    )

Only :func:`enrich_content_items` touches the optional ``everalgo.parser``
extra, and it does so lazily (inside the call). The availability helpers and
mapping never import it, so this package is safe to import without the
``everos[multimodal]`` extra installed.
"""

from .availability import has_unparsed_multimodal as has_unparsed_multimodal
from .availability import multimodal_available as multimodal_available
from .availability import require_multimodal as require_multimodal
from .enrich import enrich_content_items as enrich_content_items

__all__ = [
    "enrich_content_items",
    "has_unparsed_multimodal",
    "multimodal_available",
    "require_multimodal",
]
