"""Configuration data and Settings schema.

Public API:
    from everos.config import (
        Settings, MemorySettings, SqliteSettings, LanceDBSettings,
        LLMSettings, EmbeddingSettings, RerankSettings,
        BoundaryDetectionSettings, CascadeSettings, OmeSettings,
        load_settings, resolve_root, missing_config_error,
    )

Distinct from ``everos.component.yaml_loader`` (which is a *capability* —
loader / merger / env reader).
"""

from .hints import missing_config_error as missing_config_error
from .settings import AsyncQueueSettings as AsyncQueueSettings
from .settings import BoundaryDetectionSettings as BoundaryDetectionSettings
from .settings import CascadeSettings as CascadeSettings
from .settings import EmbeddingSettings as EmbeddingSettings
from .settings import IndexSettings as IndexSettings
from .settings import LanceDBSettings as LanceDBSettings
from .settings import LLMSettings as LLMSettings
from .settings import LoggingSettings as LoggingSettings
from .settings import MemorySettings as MemorySettings
from .settings import MilvusSettings as MilvusSettings
from .settings import MultimodalSettings as MultimodalSettings
from .settings import OmeSettings as OmeSettings
from .settings import RerankSettings as RerankSettings
from .settings import Settings as Settings
from .settings import SqliteSettings as SqliteSettings
from .settings import load_settings as load_settings
from .settings import resolve_root as resolve_root

__all__ = [
    "AsyncQueueSettings",
    "BoundaryDetectionSettings",
    "EmbeddingSettings",
    "IndexSettings",
    "LLMSettings",
    "LanceDBSettings",
    "LoggingSettings",
    "MemorySettings",
    "MilvusSettings",
    "MultimodalSettings",
    "OmeSettings",
    "RerankSettings",
    "Settings",
    "SqliteSettings",
    "load_settings",
    "missing_config_error",
    "resolve_root",
]
