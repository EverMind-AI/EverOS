"""EverOS's implementation of the everalgo ``LLMClient`` contract.

The contract and its types (``LLMClient``, ``LLMMessage``, ``LLMOutputError``,
``StrictOutputModel``) belong to everalgo; import them from ``everalgo.llm``.
This package exports only what EverOS itself defines:

- :class:`OpenAIProvider` -- the one ``LLMClient`` implementation, over any
  OpenAI-compatible Chat Completions endpoint.
- :class:`LLMCallError` -- what a failed call raises, with a ``reason``.
- :class:`LLMCaseJudge` -- everalgo's agent-case gate question answered by
  an ``LLMClient``.
- :class:`LLMCallObserver` / :class:`TokenUsage` / :class:`NullObserver` --
  the hook that sees each completion's usage and finish reason.
- :class:`LLMClientConfig` / :func:`build_llm_provider` -- the factory.
- :func:`resolve_llm_task` / :func:`resolve_llm_tasks` -- each task's
  ``[llm]`` + ``[llm.<task>]`` merge, as an :class:`LLMTaskResolution` that
  also says which section supplied each field; :func:`llm_client_config`
  turns one into a buildable config (missing credentials raise
  :class:`LLMNotConfiguredError`) and :func:`build_llm_task_providers` builds
  one provider per distinct config.
- :func:`resolve_multimodal_config` / :func:`observer_for` -- the multimodal
  section and the completion observer.

There is no process-wide accessor: the application bootstrap builds each
provider once and injects it.

External usage::

    from everos.component.llm import (
        build_llm_task_providers, llm_client_config, observer_for,
        resolve_llm_tasks,
    )

    providers = build_llm_task_providers(
        {task: llm_client_config(r) for task, r in resolve_llm_tasks(settings).items()},
        observer=observer_for(settings),
    )
"""

from .case_judge import LLMCaseJudge as LLMCaseJudge
from .errors import LLMCallError as LLMCallError
from .factory import LLMClientConfig as LLMClientConfig
from .factory import LLMNotConfiguredError as LLMNotConfiguredError
from .factory import LLMTaskResolution as LLMTaskResolution
from .factory import build_llm_provider as build_llm_provider
from .factory import build_llm_task_providers as build_llm_task_providers
from .factory import llm_client_config as llm_client_config
from .factory import observer_for as observer_for
from .factory import resolve_llm_task as resolve_llm_task
from .factory import resolve_llm_tasks as resolve_llm_tasks
from .factory import resolve_multimodal_config as resolve_multimodal_config
from .observer import LLMCallObserver as LLMCallObserver
from .observer import NullObserver as NullObserver
from .observer import TokenUsage as TokenUsage
from .openai_provider import OpenAIProvider as OpenAIProvider

__all__ = [
    "LLMCallError",
    "LLMCallObserver",
    "LLMCaseJudge",
    "LLMClientConfig",
    "LLMNotConfiguredError",
    "LLMTaskResolution",
    "NullObserver",
    "OpenAIProvider",
    "TokenUsage",
    "build_llm_provider",
    "build_llm_task_providers",
    "llm_client_config",
    "observer_for",
    "resolve_llm_task",
    "resolve_llm_tasks",
    "resolve_multimodal_config",
]
