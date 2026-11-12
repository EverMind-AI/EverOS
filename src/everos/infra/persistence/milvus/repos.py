"""Milvus repositories for EverOS derived index tables, one per table."""

from __future__ import annotations

from everos.infra.persistence.records import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    UserProfile,
)

from .repository import MilvusRepoBase


class _EpisodeRepo(MilvusRepoBase[Episode]):
    schema = Episode


class _ImageMemoryRepo(MilvusRepoBase[ImageMemory]):
    schema = ImageMemory


class _GroupProfileItemRepo(MilvusRepoBase[GroupProfileItem]):
    schema = GroupProfileItem


class _UserProfileRepo(MilvusRepoBase[UserProfile]):
    schema = UserProfile
