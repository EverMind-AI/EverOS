"""LanceDB repository for the ``image_memory`` table.

A daily-log shape like ``episode``: rows are keyed by ``entry_id`` inside one
md file and located by ``md_path``. The only thing pictures add is that the
row's vector encodes a *picture*, not the text beside it.
"""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceDailyLogRepoBase

from ...records.image_memory import ImageMemory
from ._contract import LanceIndexContract


class _ImageMemoryRepo(
    LanceIndexContract[ImageMemory], LanceDailyLogRepoBase[ImageMemory]
):
    schema = ImageMemory
