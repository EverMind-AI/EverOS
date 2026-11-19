"""LanceDB repository for the ``atomic_fact`` table."""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceDailyLogRepoBase

from ...records.atomic_fact import AtomicFact
from ._contract import LanceIndexContract


class _AtomicFactRepo(
    LanceIndexContract[AtomicFact], LanceDailyLogRepoBase[AtomicFact]
):
    schema = AtomicFact
