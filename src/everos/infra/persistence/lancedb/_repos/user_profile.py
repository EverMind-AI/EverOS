"""LanceDB repository for the ``user_profile`` table."""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceRepoBase

from ...records.user_profile import UserProfile
from ._contract import LanceIndexContract


class _UserProfileRepo(LanceIndexContract[UserProfile], LanceRepoBase[UserProfile]):
    schema = UserProfile
