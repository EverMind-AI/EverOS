"""Domain layer: the business core.

Defines memory-domain models and implements write / read / sync /
prompt management capabilities.

External usage:
    from everos.memory import (
        CanonicalMessage, IngestResult, PipelineOutcome,
        ToolCall, MemCell, Episode, AlgoMessage,
    )

Path resolution and addressing of markdown records live with the infra
writer/reader pair (``BaseDailyWriter`` / ``BaseDailyReader`` /
``SkillWriter`` / ``SkillReader`` / ``ProfileWriter`` /
``ProfileReader``) — see :mod:`everos.infra.persistence.markdown`. The
domain layer here is reserved for actual business logic (extract /
cascade / search / prompt_slots / evolution).
"""

from .models import AgentCase as AgentCase
from .models import AlgoAgentCase as AlgoAgentCase
from .models import AlgoAtomicFact as AlgoAtomicFact
from .models import AlgoEpisode as AlgoEpisode
from .models import AlgoForesight as AlgoForesight
from .models import AlgoMessage as AlgoMessage
from .models import AtomicFact as AtomicFact
from .models import CanonicalMessage as CanonicalMessage
from .models import Episode as Episode
from .models import Foresight as Foresight
from .models import IngestResult as IngestResult
from .models import MemCell as MemCell
from .models import PipelineOutcome as PipelineOutcome
from .models import ToolCall as ToolCall

__all__ = [
    "AgentCase",
    "AlgoAgentCase",
    "AlgoAtomicFact",
    "AlgoEpisode",
    "AlgoForesight",
    "AlgoMessage",
    "AtomicFact",
    "CanonicalMessage",
    "Episode",
    "Foresight",
    "IngestResult",
    "MemCell",
    "PipelineOutcome",
    "ToolCall",
]
