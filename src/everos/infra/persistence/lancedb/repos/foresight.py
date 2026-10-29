"""LanceDB repository for the ``foresight`` table."""

from __future__ import annotations

from everos.infra.persistence.lancedb.base.repository import LanceDailyLogRepoBase

from ...records.foresight import Foresight
from ._contract import LanceIndexContract


class _ForesightRepo(LanceIndexContract[Foresight], LanceDailyLogRepoBase[Foresight]):
    schema = Foresight
