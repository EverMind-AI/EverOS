"""
Content Enrich Provider

Extension point for pre-processing content items before memorization.
Enterprise can override to add multimodal parsing, content transformation, etc.

Usage pattern (same as AppLogicProvider):
- Opensource defines interface + NoopContentEnrichProvider (default)
- Enterprise implements with @component(primary=True) to auto-override via DI
"""

from abc import ABC, abstractmethod
from typing import List

from api_specs.dtos.memory import RawData
from core.di.decorators import component


class ContentEnrichProvider(ABC):
    """Content enrich provider interface.

    Pre-processes content items in RawData before memorization.
    Typical use case: multimodal file parsing (image/audio/pdf -> text).

    Extension pattern:
    - Enterprise implements this interface with @component(primary=True)
    - Automatically overrides the default NoopContentEnrichProvider via DI
    """

    @abstractmethod
    async def enrich(self, raw_data_list: List[RawData]) -> None:
        """Enrich content items in raw_data_list (in-place modification).

        Scans content items in each RawData.content["content"],
        enriches items that need processing (e.g., calling external parsing service).

        Results are written directly into content item dict fields:
        parsed_content / parsed_summary / parse_status,
        persisted to MongoDB alongside content_items.

        Args:
            raw_data_list: List of RawData to enrich (modified in-place)
        """
        ...


@component(name="content_enrich_provider")
class NoopContentEnrichProvider(ContentEnrichProvider):
    """Default no-op implementation.

    Used when opensource is deployed standalone. All non-text content remains as-is.
    """

    async def enrich(self, raw_data_list: List[RawData]) -> None:
        pass
