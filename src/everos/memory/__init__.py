"""Domain layer: the business core.

Defines memory-domain models and implements write / read / sync /
prompt management capabilities.

External usage:
    from everos.memory import (
        CanonicalMessage, IngestResult,
        ToolCall, MemCell, Episode,
    )

everalgo's own models are not re-exported: the domain converts to and
from them in :mod:`everos.memory.mappers`, and code that calls an
everalgo operator imports its types from ``everalgo`` directly.

Storage is reached only through the protocols in :mod:`everos.memory.ports`;
path resolution and addressing of markdown records live with the storage
adapters that implement them. The domain layer here is reserved for actual
business logic (extract / search / reflection / custom_instructions).
"""

from ._partition_locks import user_cluster_lock as user_cluster_lock
from .models import AgentCase as AgentCase
from .models import CanonicalMessage as CanonicalMessage
from .models import Episode as Episode
from .models import IngestResult as IngestResult
from .models import MemCell as MemCell
from .models import ToolCall as ToolCall

__all__ = [
    "AgentCase",
    "CanonicalMessage",
    "Episode",
    "IngestResult",
    "MemCell",
    "ToolCall",
    "user_cluster_lock",
]
