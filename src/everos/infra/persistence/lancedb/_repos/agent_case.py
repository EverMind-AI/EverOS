"""LanceDB repository for the ``agent_case`` table."""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceDailyLogRepoBase

from ...records.agent_case import AgentCase
from ._contract import LanceIndexContract


class _AgentCaseRepo(LanceIndexContract[AgentCase], LanceDailyLogRepoBase[AgentCase]):
    schema = AgentCase
