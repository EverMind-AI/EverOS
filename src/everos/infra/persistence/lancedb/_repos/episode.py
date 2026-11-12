"""LanceDB repository for the ``episode`` table."""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceDailyLogRepoBase

from ...records.episode import Episode
from ._contract import LanceIndexContract


class _EpisodeRepo(LanceIndexContract[Episode], LanceDailyLogRepoBase[Episode]):
    schema = Episode
