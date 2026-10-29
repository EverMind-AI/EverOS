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

from .backfill import BackfillPhase as BackfillPhase
from .backfill import BackfillPresenter as BackfillPresenter
from .backfill import NullBackfillPresenter as NullBackfillPresenter
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
from .ops import Capabilities as Capabilities
from .ops import CascadeStatus as CascadeStatus
from .ops import OmeRun as OmeRun
from .ops import OmeTriggerResult as OmeTriggerResult
from .ops import OpsService as OpsService
from .ops import QuiesceResult as QuiesceResult
from .ops import RebuildResult as RebuildResult

__all__ = [
    "TASK_STATUSES",
    "Application",
    "BackfillPhase",
    "BackfillPresenter",
    "Capabilities",
    "CascadeStatus",
    "CategoryOverview",
    "CreateDocumentResult",
    "DeleteResult",
    "DocumentContext",
    "DocumentDetail",
    "DocumentListResult",
    "DocumentNotFoundError",
    "DocumentOverviewItem",
    "DuplicateDocumentError",
    "ExtractionEmptyError",
    "KnowledgeExtractor",
    "KnowledgeService",
    "MemorizeResult",
    "MemoryService",
    "NullBackfillPresenter",
    "OmeRun",
    "OmeTriggerResult",
    "OpsService",
    "PatchResult",
    "Providers",
    "QueueHealth",
    "QuiesceResult",
    "RebuildResult",
    "SearchHit",
    "SearchKnowledgeResult",
    "TaskItem",
    "TaskPage",
    "TopicDetail",
    "TopicNotFoundError",
    "TopicOverview",
    "build_application",
    "build_providers",
]
