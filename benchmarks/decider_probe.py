"""Resolve the decider as an EverOS server does and, unless told not to, ask it once.

``run.py`` runs this in a child process whose environment is the one its EverOS
servers are launched with (``run.server_launch_env``), so ``load_settings``
reads the same variables and the same ``<EVEROS_ROOT>/everos.toml`` they do.
``resolve_llm_task`` then applies the decider's precedence -- ``[llm.decider]``,
the deprecated ``[decider]``, the decider's own defaults, ``[llm]`` -- and the
request goes through the provider ``build_llm_provider`` makes from that
resolution. Nothing is passed per call: model, endpoint, ``extra``, output cap,
temperature, timeout and SDK retries all come from the client configuration,
and the reply model makes the request carry a ``json_schema`` format, as the
multi-round decider's own call does.

Prints exactly one JSON line on stdout: ``model``, ``base_url``, ``host``,
``sources`` (field -> the section that supplied it), ``outcome``
(``resolved`` with ``--resolve-only``; otherwise ``answered``, ``failed`` or
``not_configured``), ``reason`` (the ``LLMCallError`` reason of a failed
request) and ``error``. The API key never leaves this process: it is not
printed, and it is masked out of the error text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

import structlog
from everalgo.llm import LLMMessage, StrictOutputModel

from everos.component.llm import (
    LLMCallError,
    LLMClientConfig,
    LLMNotConfiguredError,
    LLMTaskResolution,
    build_llm_provider,
    llm_client_config,
    resolve_llm_task,
)
from everos.config import LLMTask, load_settings

_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ready": true}'
"""Spelled out because ``json_object`` and ``prompt_only`` endpoints get no schema."""

_MASK = "***"


class DeciderProbeReply(StrictOutputModel):
    """The probe's reply model; any reply that validates against it is an answer."""

    ready: bool


def main(argv: Sequence[str] | None = None) -> None:
    """Print the decider's resolution and, unless ``--resolve-only``, its answer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="resolve the decider without sending it a request",
    )
    args = parser.parse_args(argv)
    # The provider's warnings would otherwise share stdout with the report.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    print(json.dumps(probe(send=not args.resolve_only)))


def probe(*, send: bool) -> dict[str, Any]:
    """The decider's resolution under this process's environment.

    Args:
        send: Also send the decider one request and report how it went.

    Returns:
        The report the module docstring describes.

    Raises:
        pydantic.ValidationError: The environment or ``everos.toml`` holds a
            setting EverOS rejects; the process then exits without a report.
    """
    resolution = resolve_llm_task(load_settings(), LLMTask.DECIDER)
    outcome = (
        _ask(resolution) if send else {"outcome": "resolved", "reason": "", "error": ""}
    )
    return {**_describe(resolution), **outcome}


def _describe(resolution: LLMTaskResolution) -> dict[str, Any]:
    """Everything the report says about the resolution; never the key."""
    return {
        "model": resolution.settings.model,
        "base_url": resolution.settings.base_url or "",
        "host": resolution.host or "",
        "sources": dict(resolution.sources),
    }


def _ask(resolution: LLMTaskResolution) -> dict[str, str]:
    """Send the one request; how it went is reported, not raised."""
    try:
        config = llm_client_config(resolution)
    except LLMNotConfiguredError as exc:
        return {"outcome": "not_configured", "reason": "", "error": str(exc)}
    try:
        asyncio.run(_send(config))
    except LLMCallError as exc:
        return {
            "outcome": "failed",
            "reason": exc.reason,
            "error": _masked(str(exc), config.api_key),
        }
    return {"outcome": "answered", "reason": "", "error": ""}


async def _send(config: LLMClientConfig) -> None:
    """One ``achat`` through the provider the server would build from ``config``."""
    await build_llm_provider(config).achat(
        [LLMMessage(role="user", content=_PROMPT)],
        response_format=DeciderProbeReply,
    )


def _masked(text: str, secret: str) -> str:
    """``text`` with every occurrence of ``secret`` replaced by a mask."""
    return text.replace(secret, _MASK) if secret else text


if __name__ == "__main__":
    main()
