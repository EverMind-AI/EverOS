"""Application layer: assembly plus two use-case facades.

:func:`build_application` builds an :class:`Application` from settings; an
entry point holds that one object and calls its facades --
:class:`MemoryService` (add / flush / search / get) and
:class:`OpsService` (cascade, OME, health, tasks). One CLI command or API
endpoint maps to one facade method. :func:`prompt_manifest` names the prompts
this EverOS sends (``everos prompts``).

External usage::

    from everos.config import load_settings
    from everos.service import build_application

    application = build_application(load_settings())
    await application.start()            # a server; a CLI command skips this
    result = await application.memory.add(
        payload, request_id="req-1", async_mode=False, defer_extraction=False
    )
    await application.aclose()
"""

from everos.memory.get import GetRequest as GetRequest
from everos.memory.get import GetResponse as GetResponse
from everos.memory.prompt_manifest import prompt_manifest as prompt_manifest
from everos.memory.search import NO_AGENT_MEMORY as NO_AGENT_MEMORY
from everos.memory.search import GroupProfileEntry as GroupProfileEntry
from everos.memory.search import SearchRequest as SearchRequest
from everos.memory.search import SearchResponse as SearchResponse

from .bootstrap import Application as Application
from .bootstrap import Providers as Providers
from .bootstrap import build_application as build_application
from .bootstrap import build_providers as build_providers
from .bootstrap import configure_process as configure_process
from .memory import TASK_STATUSES as TASK_STATUSES
from .memory import FlushResult as FlushResult
from .memory import GroupForgetResult as GroupForgetResult
from .memory import GroupForgetTarget as GroupForgetTarget
from .memory import GroupUndoResult as GroupUndoResult
from .memory import GroupUpsertResult as GroupUpsertResult
from .memory import GroupView as GroupView
from .memory import MemorizeResult as MemorizeResult
from .memory import MemoryChanges as MemoryChanges
from .memory import MemoryHistory as MemoryHistory
from .memory import MemoryService as MemoryService
from .memory import MemoryVersion as MemoryVersion
from .memory import ProjectMember as ProjectMember
from .memory import ProjectProfile as ProjectProfile
from .memory import ProjectView as ProjectView
from .memory import Queued as Queued
from .memory import QueueHealth as QueueHealth
from .memory import RosterChangeResult as RosterChangeResult
from .memory import RosterEventInput as RosterEventInput
from .memory import TaskItem as TaskItem
from .memory import TaskPage as TaskPage
from .memory import UpdateResult as UpdateResult
from .memory import UserForgetResult as UserForgetResult
from .memory import UserForgetTarget as UserForgetTarget
from .memory import UserUndoResult as UserUndoResult
from .ops import BACKFILL_PHASES as BACKFILL_PHASES
from .ops import BackfillDeps as BackfillDeps
from .ops import BackfillOutcome as BackfillOutcome
from .ops import BackfillPresenter as BackfillPresenter
from .ops import BackfillSummary as BackfillSummary
from .ops import Capabilities as Capabilities
from .ops import CascadeHealthReport as CascadeHealthReport
from .ops import CascadeStatus as CascadeStatus
from .ops import HealthReport as HealthReport
from .ops import NullBackfillPresenter as NullBackfillPresenter
from .ops import OmeRun as OmeRun
from .ops import OmeTriggerResult as OmeTriggerResult
from .ops import OpsService as OpsService
from .ops import ProviderStatus as ProviderStatus
from .ops import QuiesceResult as QuiesceResult
from .ops import RebuildResult as RebuildResult
from .ops import VectorsPhaseResult as VectorsPhaseResult
from .ops import compute_disabled_features as compute_disabled_features
from .ops import format_tokens as format_tokens

__all__ = [
    "BACKFILL_PHASES",
    "NO_AGENT_MEMORY",
    "TASK_STATUSES",
    "Application",
    "BackfillDeps",
    "BackfillOutcome",
    "BackfillPresenter",
    "BackfillSummary",
    "Capabilities",
    "CascadeHealthReport",
    "CascadeStatus",
    "FlushResult",
    "GetRequest",
    "GetResponse",
    "GroupForgetResult",
    "GroupForgetTarget",
    "GroupProfileEntry",
    "GroupUndoResult",
    "GroupUpsertResult",
    "GroupView",
    "HealthReport",
    "MemorizeResult",
    "MemoryChanges",
    "MemoryHistory",
    "MemoryService",
    "MemoryVersion",
    "NullBackfillPresenter",
    "OmeRun",
    "OmeTriggerResult",
    "OpsService",
    "ProjectMember",
    "ProjectProfile",
    "ProjectView",
    "ProviderStatus",
    "Providers",
    "QueueHealth",
    "Queued",
    "QuiesceResult",
    "RebuildResult",
    "RosterChangeResult",
    "RosterEventInput",
    "SearchRequest",
    "SearchResponse",
    "TaskItem",
    "TaskPage",
    "UpdateResult",
    "UserForgetResult",
    "UserForgetTarget",
    "UserUndoResult",
    "VectorsPhaseResult",
    "build_application",
    "build_providers",
    "compute_disabled_features",
    "configure_process",
    "format_tokens",
    "prompt_manifest",
]
