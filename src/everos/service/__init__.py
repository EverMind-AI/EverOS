"""Application layer: assembly plus three use-case facades.

:func:`build_application` builds an :class:`Application` from settings; an
entry point holds that one object and calls its facades --
:class:`MemoryService` (memorize / flush / search / get),
:class:`KnowledgeService` (knowledge documents and search) and
:class:`OpsService` (cascade, OME, health, tasks). One CLI command or API
endpoint maps to one facade method.

External usage::

    from everos.config import load_settings
    from everos.service import build_application

    application = build_application(load_settings())
    await application.start()            # a server; a CLI command skips this
    result = await application.memory.memorize(payload)
    await application.aclose()
"""

from everos.core.errors import DocumentNotFoundError as DocumentNotFoundError
from everos.core.errors import DuplicateDocumentError as DuplicateDocumentError
from everos.core.errors import ExtractionEmptyError as ExtractionEmptyError
from everos.core.errors import TopicNotFoundError as TopicNotFoundError
from everos.memory.get import GetRequest as GetRequest
from everos.memory.get import GetResponse as GetResponse
from everos.memory.search import SearchRequest as SearchRequest
from everos.memory.search import SearchResponse as SearchResponse

from .bootstrap import Application as Application
from .bootstrap import Providers as Providers
from .bootstrap import build_application as build_application
from .bootstrap import build_providers as build_providers
from .knowledge import CategoryOverview as CategoryOverview
from .knowledge import CreateDocumentResult as CreateDocumentResult
from .knowledge import DeleteResult as DeleteResult
from .knowledge import DocumentContext as DocumentContext
from .knowledge import DocumentDetail as DocumentDetail
from .knowledge import DocumentListResult as DocumentListResult
from .knowledge import DocumentOverviewItem as DocumentOverviewItem
from .knowledge import KnowledgeExtractor as KnowledgeExtractor
from .knowledge import KnowledgeService as KnowledgeService
from .knowledge import PatchResult as PatchResult
from .knowledge import SearchHit as SearchHit
from .knowledge import SearchKnowledgeResult as SearchKnowledgeResult
from .knowledge import TopicDetail as TopicDetail
from .knowledge import TopicOverview as TopicOverview
from .memorize_queue import TASK_STATUSES as TASK_STATUSES
from .memorize_queue import QueueHealth as QueueHealth
from .memorize_queue import TaskItem as TaskItem
from .memorize_queue import TaskPage as TaskPage
from .memory import MemorizeResult as MemorizeResult
from .memory import MemoryService as MemoryService
from .memory import Queued as Queued
from .ops import BACKFILL_PHASES as BACKFILL_PHASES
from .ops import BackfillDeps as BackfillDeps
from .ops import BackfillOutcome as BackfillOutcome
from .ops import BackfillPresenter as BackfillPresenter
from .ops import BackfillSummary as BackfillSummary
from .ops import Capabilities as Capabilities
from .ops import CascadeHealthReport as CascadeHealthReport
from .ops import CascadeStatus as CascadeStatus
from .ops import ClustersPhaseResult as ClustersPhaseResult
from .ops import HealthReport as HealthReport
from .ops import NullBackfillPresenter as NullBackfillPresenter
from .ops import OmeRun as OmeRun
from .ops import OmeTriggerResult as OmeTriggerResult
from .ops import OpsService as OpsService
from .ops import ProviderStatus as ProviderStatus
from .ops import QuiesceResult as QuiesceResult
from .ops import RebuildResult as RebuildResult
from .ops import SkillsPhaseResult as SkillsPhaseResult
from .ops import VectorsPhaseResult as VectorsPhaseResult
from .ops import compute_disabled_features as compute_disabled_features
from .ops import format_tokens as format_tokens

__all__ = [
    "BACKFILL_PHASES",
    "TASK_STATUSES",
    "Application",
    "BackfillDeps",
    "BackfillOutcome",
    "BackfillPresenter",
    "BackfillSummary",
    "Capabilities",
    "CascadeHealthReport",
    "CascadeStatus",
    "CategoryOverview",
    "ClustersPhaseResult",
    "CreateDocumentResult",
    "DeleteResult",
    "DocumentContext",
    "DocumentDetail",
    "DocumentListResult",
    "DocumentNotFoundError",
    "DocumentOverviewItem",
    "DuplicateDocumentError",
    "ExtractionEmptyError",
    "GetRequest",
    "GetResponse",
    "HealthReport",
    "KnowledgeExtractor",
    "KnowledgeService",
    "MemorizeResult",
    "MemoryService",
    "NullBackfillPresenter",
    "OmeRun",
    "OmeTriggerResult",
    "OpsService",
    "PatchResult",
    "ProviderStatus",
    "Providers",
    "QueueHealth",
    "Queued",
    "QuiesceResult",
    "RebuildResult",
    "SearchHit",
    "SearchKnowledgeResult",
    "SearchRequest",
    "SearchResponse",
    "SkillsPhaseResult",
    "TaskItem",
    "TaskPage",
    "TopicDetail",
    "TopicNotFoundError",
    "TopicOverview",
    "VectorsPhaseResult",
    "build_application",
    "build_providers",
    "compute_disabled_features",
    "format_tokens",
]
