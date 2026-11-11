"""LanceDB repository for the ``group_profile_item`` table.

A daily-log shape like ``episode``: rows are keyed by ``entry_id`` inside one
md file and located by ``md_path``. The table carries no vector, so it only
ever answers scalar filters and BM25 over ``statement_tokens``; it has no
``session_id`` / ``parent_*`` columns, so the chassis' ``find_by_session`` /
``find_by_parent`` do not apply to it.
"""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceDailyLogRepoBase

from ...records.group_profile_item import GroupProfileItem
from ._contract import LanceIndexContract


class _GroupProfileItemRepo(
    LanceIndexContract[GroupProfileItem], LanceDailyLogRepoBase[GroupProfileItem]
):
    schema = GroupProfileItem
